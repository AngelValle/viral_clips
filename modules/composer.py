"""
composer.py — v6

Cambios vs v5:
  - Overlay neón: MOV qtrle animado → PNG RGBA estático con -loop 1
    El MOV tardaba 5+ min porque qtrle no tiene aceleración hardware.
    El PNG con -loop 1 lo maneja FFmpeg como imagen estática, sin coste.
  - hwaccel cuda: NVDEC para decodificación GPU, compatible con filtros CPU.
  - -threads 0: FFmpeg usa todos los cores disponibles para filtros CPU.
  - Eliminado -stream_loop -1 (causaba seek ineficiente sobre qtrle).

Layout con cámara:
  - Gameplay fullscreen (fondo)
  - Webcam con su borde morado original como overlay flotante
  - Nombre del streamer en la parte inferior del recuadro webcam

Layout sin cámara:
  - Gameplay fullscreen
  - Nombre del streamer centrado en la parte superior
"""

import json
import logging
import os
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from modules.gpu_utils import get_ffmpeg_hwaccel

logger = logging.getLogger(__name__)


# ── Neon name animation (inlineado desde neon_name.py) ───────────────────────

def _find_font(font_name: str) -> Optional[str]:
    import sys, glob
    name_lower = font_name.lower().replace(" ", "")
    if sys.platform == "win32":
        win_roots = [r"C:\Windows\Fonts", os.path.expanduser(r"~\AppData\Local\Microsoft\Windows\Fonts")]
        for root in win_roots:
            if not os.path.isdir(root):
                continue
            for fname in os.listdir(root):
                if fname.lower().endswith((".ttf", ".otf")) and name_lower in fname.lower().replace(" ", "").replace("-", ""):
                    return os.path.join(root, fname)
        for fallback in ["arialbd.ttf", "calibrib.ttf", "verdanab.ttf"]:
            path = os.path.join(r"C:\Windows\Fonts", fallback)
            if os.path.exists(path):
                return path
        return None
    for pattern in ["/usr/share/fonts/**/*.ttf", os.path.expanduser("~/.fonts/**/*.ttf")]:
        for path in glob.glob(pattern, recursive=True):
            if name_lower in os.path.basename(path).lower().replace(" ", "").replace("-", ""):
                return path
    return None


def _build_static_layers(
    text: str, width: int, height: int, font: Any, neon_rgb: Tuple[int, int, int]
) -> Tuple[np.ndarray, np.ndarray]:
    from PIL import Image, ImageDraw, ImageFilter
    nr, ng, nb = neon_rgb
    probe = ImageDraw.Draw(Image.new("RGBA", (width, height)))
    bbox  = probe.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    tx, ty  = (width - tw) // 2, (height - th) // 2

    fill_img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    for radius, alpha in [(14, 180), (7, 140), (3, 100)]:
        layer = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        ImageDraw.Draw(layer).text((tx, ty), text, font=font, fill=(nr, ng, nb, alpha))
        fill_img = Image.alpha_composite(fill_img, layer.filter(ImageFilter.GaussianBlur(radius=radius)))

    d = ImageDraw.Draw(fill_img)
    for dx, dy in [(-2, 0), (2, 0), (0, -2), (0, 2)]:
        d.text((tx + dx, ty + dy), text, font=font, fill=(0, 0, 0, 60))
    d.text((tx, ty), text, font=font, fill=(nr, ng, nb, 255))

    stroke_img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    ImageDraw.Draw(stroke_img).text((tx, ty), text, font=font, fill=(0, 0, 0, 0), stroke_width=2, stroke_fill=(255, 255, 255, 255))

    return np.array(fill_img, dtype=np.uint8), np.array(stroke_img, dtype=np.float32)[:, :, 3] / 255.0


def _render_frame(fill_arr: np.ndarray, stroke_mask: np.ndarray, width: int, height: int, t: float) -> Any:
    from PIL import Image, ImageFilter
    sweep_x = t * (width + width * 0.2) - width * 0.1
    sweep_w, trail_w = width * 0.15, width * 0.30

    xs  = np.arange(width, dtype=np.float32)
    haz = np.exp(-((xs - sweep_x) ** 2) / (2 * (sweep_w * 0.4) ** 2))
    estela = np.where(xs < sweep_x, np.clip((1.0 - (sweep_x - xs) / trail_w), 0, 1) ** 2, 0.0)
    intensity_2d = np.tile(np.clip(0.15 + estela * 0.5 + haz * 1.0, 0, 1), (height, 1))

    stroke_frame = np.zeros((height, width, 4), dtype=np.uint8)
    stroke_frame[:, :, :3] = 255
    stroke_frame[:, :, 3] = (stroke_mask * intensity_2d * 255).clip(0, 255).astype(np.uint8)

    halo_arr = np.zeros((height, width, 4), dtype=np.uint8)
    halo_arr[:, :, :3] = 255
    halo_arr[:, :, 3] = (stroke_mask * np.tile(haz, (height, 1)) * 200).clip(0, 255).astype(np.uint8)
    halo = Image.fromarray(halo_arr, "RGBA").filter(ImageFilter.GaussianBlur(radius=6))

    result = Image.fromarray(fill_arr, "RGBA")
    result = Image.alpha_composite(result, halo)
    return Image.alpha_composite(result, Image.fromarray(stroke_frame, "RGBA"))


def generate_neon_overlay(
    text: str, width: int, height: int, fps: int, output_path: Path, config: dict
) -> Path:
    from PIL import ImageFont
    layout      = config.get("layout", {})
    sweep_speed = layout.get("neon_sweep_speed", 0.6)
    font_name   = config.get("subtitles", {}).get("font_name", "Showcard Gothic")
    font_size   = max(12, int(height * layout.get("neon_font_size_ratio", 0.65)))
    bgr         = layout.get("webcam_name_color_bgr", [255, 50, 255])

    font_path    = _find_font(font_name)
    font         = ImageFont.truetype(font_path, font_size) if font_path else ImageFont.load_default()
    cycle_frames = max(10, int(round(fps / sweep_speed)))
    fill_arr, stroke_mask = _build_static_layers(
        text, width, height, font, (int(bgr[2]), int(bgr[1]), int(bgr[0]))
    )

    with tempfile.TemporaryDirectory() as td:
        tdp = Path(td)

        def render_and_save(i: int) -> None:
            _render_frame(fill_arr, stroke_mask, width, height, i / cycle_frames).save(
                str(tdp / f"f{i:04d}.png")
            )

        with ThreadPoolExecutor(max_workers=min(os.cpu_count() or 4, 8)) as executor:
            list(executor.map(render_and_save, range(cycle_frames)))

        r = subprocess.run([
            "ffmpeg", "-y", "-framerate", str(fps), "-i", str(tdp / "f%04d.png"),
            "-vcodec", "png", "-pix_fmt", "rgba", str(output_path),
        ], capture_output=True, timeout=60)

        if r.returncode != 0:
            raise RuntimeError("Fallo crítico armando la animación de Neón.")

    return output_path


# ── Caché de MOV neón ─────────────────────────────────────────────────────────

def _neon_mov_cached(
    text: str, width: int, height: int,
    fps: int, config: dict,
    cache_dir: Path,
    variant: str,          # "cam" o "top"
) -> Path:
    """
    Devuelve la ruta del MOV neón, generándolo solo si no existe
    o si los parámetros han cambiado.

    Clave de caché: nombre + dimensiones + fps + pulse_speed + color + fuente.
    """
    import hashlib, json as _json

    layout     = config.get("layout", {})
    font_name  = config.get("subtitles", {}).get("font_name", "Showcard Gothic")
    key_data   = {
        "text":        text,
        "width":       width,
        "height":      height,
        "fps":         fps,
        "pulse_speed": layout.get("neon_pulse_speed", 1.5),
        "color_bgr":   layout.get("webcam_name_color_bgr", [255, 50, 255]),
        "font":        font_name,
        "font_ratio":  layout.get("neon_font_size_ratio", 0.65),
    }
    key_hash  = hashlib.md5(_json.dumps(key_data, sort_keys=True).encode()).hexdigest()[:12]
    mov_path  = cache_dir / f"neon_{variant}_{key_hash}.mov"

    if mov_path.exists() and mov_path.stat().st_size > 0:
        logger.info(f"Overlay neón: {mov_path.name} ↩ caché")
        return mov_path

    # Generar y guardar en caché
    generate_neon_overlay(text, width, height, fps, mov_path, config)
    return mov_path


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_video_info(video_path: Path) -> Tuple[int, int, float]:
    result = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json",
         "-show_streams", "-select_streams", "v:0", str(video_path)],
        capture_output=True, text=True,
    )
    try:
        s = json.loads(result.stdout)["streams"][0]
        num, den = s["r_frame_rate"].split("/")
        return int(s["width"]), int(s["height"]), float(num) / float(den)
    except Exception:
        return 2560, 1440, 60.0


def _run_ffmpeg(cmd: list) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)


def _even(n: int) -> int:
    return n if n % 2 == 0 else n - 1


# ── Crops ─────────────────────────────────────────────────────────────────────

def _gameplay_crop(src_w: int, src_h: int, layout: dict,
                   zoom: Optional[float] = None,
                   is_driving: bool = False) -> str:
    z        = zoom or layout.get("gameplay_zoom", 2.63)
    skip_y   = layout.get("stats_skip_ratio", 0.0)
    usable_h = int(src_h * (1.0 - skip_y))
    usable_y = int(src_h * skip_y)
    crop_h   = int(usable_h / z)
    crop_w   = int(src_w / z)
    if is_driving:
        x_offset = layout.get("gameplay_x_offset_driving", -150)
    else:
        x_offset = layout.get("gameplay_x_offset", -200)
    crop_x   = (src_w - crop_w) // 2 + x_offset
    crop_y   = usable_y + (usable_h - crop_h) // 2
    crop_x   = max(0, min(crop_x, src_w - crop_w))
    crop_y   = max(0, min(crop_y, src_h - crop_h))
    return f"crop={crop_w}:{crop_h}:{crop_x}:{crop_y}"


def _scale_fill(tw: int, th: int) -> str:
    tw = _even(tw)
    th = _even(th)
    return (
        f"scale={tw}:{th}:force_original_aspect_ratio=increase,"
        f"scale=trunc(iw/2)*2:trunc(ih/2)*2,"
        f"crop={tw}:{th}:(iw-{tw})/2:(ih-{th})/2,"
        f"setsar=1"
    )


def _webcam_crop(src_w: int, src_h: int, layout: dict) -> Tuple[str, int, int, int, int]:
    w_ratio  = layout.get("webcam_w_ratio",  0.137)
    h_ratio  = layout.get("webcam_h_ratio",  0.331)
    x_offset = layout.get("webcam_x_offset", 22)
    cam_w    = int(src_w * w_ratio)
    cam_h    = int(src_h * h_ratio)
    cam_x    = x_offset
    if "webcam_y_center_ratio" in layout:
        cam_y = int(src_h * layout["webcam_y_center_ratio"]) - cam_h // 2
    else:
        cam_y = src_h - cam_h - layout.get("webcam_y_offset", 0)
    cam_x = max(0, min(cam_x, src_w - cam_w))
    cam_y = max(0, min(cam_y, src_h - cam_h))
    return f"crop={cam_w}:{cam_h}:{cam_x}:{cam_y}", cam_x, cam_y, cam_w, cam_h


# ── Intervalos con cámara ─────────────────────────────────────────────────────

def _face_intervals(face_data: List[Dict], video_fps: float,
                    min_sec: float) -> List[Tuple[float, float]]:
    if not face_data:
        return []

    n          = len(face_data)
    min_frames = max(1, int(min_sec * video_fps))
    flags      = [d["has_face"] for d in face_data]

    changed = True
    while changed:
        changed = False
        i = 0
        while i < n:
            val = flags[i]
            j   = i
            while j < n and flags[j] == val:
                j += 1
            if (j - i) < min_frames:
                prev = flags[i - 1] if i > 0 else (not val)
                for k in range(i, j):
                    flags[k] = prev
                changed = True
            i = j

    intervals = []
    i = 0
    while i < n:
        if flags[i]:
            j = i
            while j < n and flags[j]:
                j += 1
            intervals.append((round(i / video_fps, 4), round(j / video_fps, 4)))
            i = j
        else:
            i += 1
    return intervals


# ── Filtergraphs ──────────────────────────────────────────────────────────────
# Input 0 = vídeo fuente
# Input 1 = PNG neón (imagen estática, -loop 1)

def _filtergraph_with_cam(
    src_w: int, src_h: int, res_w: int, res_h: int,
    layout: dict,
    intervals: List[Tuple[float, float]],
    cam_crop: str, cam_w: int, cam_h: int,
    name_h: int,
    zoom_fs: float,
    is_driving: bool = False,
) -> str:
    ow     = _even(int(res_w * layout.get("webcam_overlay_w_ratio", 0.485)))
    oh     = _even(int(ow * cam_h / cam_w))
    dest_x = (res_w - ow) // 2
    dest_y = int(res_h * layout.get("webcam_overlay_y_ratio", 0.01))
    name_x = dest_x
    name_y = dest_y + oh - name_h

    between = "+".join(f"between(t,{s},{e})" for s, e in intervals)
    enable  = f"'gte({between},1)'"
    gp_crop = _gameplay_crop(src_w, src_h, layout, zoom=zoom_fs, is_driving=is_driving)

    return ";".join([
        f"[0:v]split=2[raw_bg][raw_cam]",
        f"[raw_bg]{gp_crop},{_scale_fill(res_w, res_h)}[base]",
        f"[raw_cam]{cam_crop},scale={ow}:{oh}[cam_scaled]",
        f"[base][cam_scaled]overlay={dest_x}:{dest_y}:enable={enable}[with_cam]",
        f"[with_cam][1:v]overlay={name_x}:{name_y}:enable={enable}",
    ])


def _filtergraph_no_cam(
    src_w: int, src_h: int, res_w: int, res_h: int,
    layout: dict,
    zoom_fs: float,
    is_driving: bool = False,
) -> str:
    gp_crop = _gameplay_crop(src_w, src_h, layout, zoom=zoom_fs, is_driving=is_driving)
    name_y  = int(res_h * layout.get("name_top_y_ratio", 0.02))

    return ";".join([
        f"[0:v]{gp_crop},{_scale_fill(res_w, res_h)}[base]",
        f"[base][1:v]overlay=0:{name_y}",
    ])


# ── Punto de entrada ──────────────────────────────────────────────────────────

def compose_dynamic(
    raw_clip_path: Path,
    face_data: List[Dict],
    config: dict,
    output_path: Path,
    is_driving: bool = False,
) -> Path:
    cfg_o         = config["output"]
    layout        = config.get("layout", {})
    res_w         = cfg_o["resolution_w"]
    res_h         = cfg_o["resolution_h"]
    fps           = cfg_o["fps"]
    streamer_name = config.get("claude", {}).get("streamer_name", "")
    zoom_fs       = layout.get("fullscreen_zoom", 3.5)

    src_w, src_h, video_fps = _get_video_info(raw_clip_path)
    logger.info(f"Fuente: {src_w}x{src_h}@{video_fps:.0f}fps → {res_w}x{res_h}")

    intervals = _face_intervals(face_data, video_fps, layout.get("min_segment_sec", 0.5))
    if intervals:
        logger.info(f"Cámara en {len(intervals)} intervalo(s): "
                    + ", ".join(f"{s:.1f}–{e:.1f}s" for s, e in intervals))
    else:
        logger.info("Sin cámara → gameplay fullscreen + nombre arriba")

    hw = get_ffmpeg_hwaccel()
    logger.info(f"Encoder: {hw['encoder']} ({'GPU' if hw['is_hw'] else 'CPU'}) | "
                f"conducción: {'SÍ → x_offset=-150' if is_driving else 'NO → x_offset=-200'}")

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)

        cam_crop_str, _, _, cam_w, cam_h = _webcam_crop(src_w, src_h, layout)
        ow = _even(int(res_w * layout.get("webcam_overlay_w_ratio", 0.485)))
        oh = _even(int(ow * cam_h / cam_w))

        # Duración real del clip (necesaria para -t en el input PNG)
        try:
            import subprocess as _sp
            _probe = _sp.run(
                ["ffprobe", "-v", "quiet", "-print_format", "json",
                 "-show_format", str(raw_clip_path)],
                capture_output=True, text=True,
            )
            clip_dur = float(json.loads(_probe.stdout)["format"]["duration"])
        except Exception:
            clip_dur = float(config["viral_detection"].get("max_clip_duration", 60.0))

        # MOV neón: se genera una vez y se reutiliza desde caché.
        # Clave: nombre + dimensiones + fps + color + fuente.
        neon_cache_dir = Path(config["paths"].get("cache_dir", "videos/cache")) / "neon"
        neon_cache_dir.mkdir(parents=True, exist_ok=True)

        if streamer_name:
            if intervals:
                name_h   = _even(int(oh * layout.get("webcam_name_h_ratio", 0.10)))
                name_mov = _neon_mov_cached(streamer_name, ow, name_h, fps, config,
                                            neon_cache_dir, "cam")
            else:
                name_h   = _even(int(res_h * layout.get("name_top_h_ratio", 0.06)))
                name_mov = _neon_mov_cached(streamer_name, res_w, name_h, fps, config,
                                            neon_cache_dir, "top")
        else:
            # Sin nombre: MOV vacío 2x2 de 1 frame (también cacheado)
            name_mov = neon_cache_dir / "neon_empty.mov"
            if not name_mov.exists():
                subprocess.run([
                    "ffmpeg", "-y", "-f", "lavfi",
                    "-i", "color=black@0:size=2x2:duration=0.1:rate=1",
                    "-vcodec", "png", "-pix_fmt", "rgba", str(name_mov)
                ], capture_output=True)
            name_h = 0

        if intervals:
            vf = _filtergraph_with_cam(
                src_w, src_h, res_w, res_h, layout,
                intervals, cam_crop_str, cam_w, cam_h,
                name_h, zoom_fs, is_driving=is_driving,
            )
        else:
            vf = _filtergraph_no_cam(src_w, src_h, res_w, res_h, layout, zoom_fs,
                                     is_driving=is_driving)

        def build_cmd(encoder: str, extra_args: list) -> list:
            # -hwaccel cuda activa NVDEC (decodificación en GPU).
            # Los frames se descargan a RAM automáticamente antes del filtergraph,
            # compatible con todos los filtros CPU (crop, scale, overlay).
            # La ganancia principal es NVENC en el encoding de salida.
            use_nvdec = hw["hwaccel"] == "cuda" and encoder == hw["encoder"]
            cmd = ["ffmpeg", "-y", "-threads", "0"]
            if use_nvdec:
                cmd += ["-hwaccel", "cuda"]
            cmd += [
                "-i", str(raw_clip_path),
                # -stream_loop -1: repite el ciclo MOV en bucle.
                # -t clip_dur: detiene el loop cuando termina el vídeo fuente.
                "-stream_loop", "-1", "-t", str(round(clip_dur, 3)), "-i", str(name_mov),
                "-filter_complex", vf,
                "-r", str(fps),
                "-c:v", encoder, *extra_args,
                "-c:a", "copy",
                "-avoid_negative_ts", "make_zero",
                "-movflags", "+faststart",
                str(output_path),
            ]
            return cmd

        result = _run_ffmpeg(build_cmd(hw["encoder"], hw["extra_enc_args"]))
        if result.returncode != 0:
            err = result.stderr.decode(errors="ignore")
            logger.warning(f"Encoder {hw['encoder']} falló, reintentando con libx264:\n{err[-400:]}")
            result = _run_ffmpeg(build_cmd("libx264", ["-preset", "fast", "-crf", "23"]))

        if result.returncode != 0:
            err = result.stderr.decode(errors="ignore")
            logger.error(f"FFmpeg error:\n{err[-800:]}")
            raise RuntimeError("Fallo en la composición del vídeo")

    logger.info(f"Compuesto: {output_path.name}")
    return output_path
