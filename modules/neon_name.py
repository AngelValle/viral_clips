"""
neon_name.py
Genera un MOV de UN ciclo de animación "trazo infinito" para el nombre del streamer.

Efecto:
  - Letras en color neón con halo difuso (estático)
  - Contorno blanco animado que barre las letras de izquierda a derecha en loop
  - Estela del contorno que se apaga suavemente detrás del haz
  - Halo blanco difuso alrededor del punto brillante del haz

Caché: el MOV se genera una vez y se reutiliza. La clave incluye texto,
dimensiones, fps, velocidad, color y fuente.
"""

import logging
import os
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def _find_font(font_name: str) -> Optional[str]:
    import sys, glob
    if sys.platform == "win32":
        win_roots = [
            r"C:\Windows\Fonts",
            os.path.expanduser(r"~\AppData\Local\Microsoft\Windows\Fonts"),
        ]
        name_lower = font_name.lower().replace(" ", "")
        for root in win_roots:
            if not os.path.isdir(root):
                continue
            for fname in os.listdir(root):
                if fname.lower().endswith((".ttf", ".otf")):
                    stem = fname.lower().replace(" ", "").replace("-", "")
                    if name_lower in stem:
                        return os.path.join(root, fname)
        for fallback in ["arialbd.ttf", "calibrib.ttf", "verdanab.ttf", "arial.ttf"]:
            path = os.path.join(r"C:\Windows\Fonts", fallback)
            if os.path.exists(path):
                return path
        return None
    name_lower = font_name.lower().replace(" ", "")
    for pattern in ["/usr/share/fonts/**/*.ttf", "/usr/local/share/fonts/**/*.ttf",
                    os.path.expanduser("~/.fonts/**/*.ttf")]:
        for path in glob.glob(pattern, recursive=True):
            stem = os.path.basename(path).lower().replace(" ", "").replace("-", "")
            if name_lower in stem:
                return path
    hits = glob.glob("/usr/share/fonts/**/*.ttf", recursive=True)
    return hits[0] if hits else None


def _build_static_layers(
    text: str, width: int, height: int,
    font, neon_rgb: Tuple[int, int, int]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Precalcula dos arrays estáticos reutilizados en cada frame:
      fill_arr   : relleno neón RGBA (uint8) — estático, no cambia
      stroke_mask: máscara float32 del contorno blanco — modulada por el barrido
    """
    from PIL import Image, ImageDraw, ImageFilter

    nr, ng, nb = neon_rgb

    probe = ImageDraw.Draw(Image.new("RGBA", (width, height)))
    bbox  = probe.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    tx = (width  - tw) // 2
    ty = (height - th) // 2

    # ── Relleno neón: halo difuso + texto sólido ──────────────────────────────
    fill_img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    for radius, alpha in [(14, 180), (7, 140), (3, 100)]:
        layer = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        ImageDraw.Draw(layer).text((tx, ty), text, font=font, fill=(nr, ng, nb, alpha))
        fill_img = Image.alpha_composite(
            fill_img, layer.filter(ImageFilter.GaussianBlur(radius=radius))
        )
    d = ImageDraw.Draw(fill_img)
    for dx, dy in [(-2,0),(2,0),(0,-2),(0,2)]:
        d.text((tx+dx, ty+dy), text, font=font, fill=(0, 0, 0, 180))
    d.text((tx, ty), text, font=font, fill=(nr, ng, nb, 255))

    # ── Contorno blanco: stroke_width nativo de PIL ───────────────────────────
    stroke_img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    ImageDraw.Draw(stroke_img).text(
        (tx, ty), text, font=font,
        fill=(0, 0, 0, 0),
        stroke_width=2,
        stroke_fill=(255, 255, 255, 255),
    )

    fill_arr   = np.array(fill_img, dtype=np.uint8)
    stroke_mask = np.array(stroke_img, dtype=np.float32)[:, :, 3] / 255.0
    return fill_arr, stroke_mask


def _render_frame(
    fill_arr: np.ndarray,
    stroke_mask: np.ndarray,
    width: int,
    height: int,
    t: float,   # 0.0 → 1.0
) -> "PIL.Image.Image":
    from PIL import Image, ImageFilter

    sweep_x = t * (width + width * 0.2) - width * 0.1
    sweep_w = width * 0.15
    trail_w = width * 0.30

    xs = np.arange(width, dtype=np.float32)
    haz    = np.exp(-((xs - sweep_x) ** 2) / (2 * (sweep_w * 0.4) ** 2))
    estela = np.where(
        xs < sweep_x,
        np.clip((1.0 - (sweep_x - xs) / trail_w), 0, 1) ** 2,
        0.0,
    )
    intensity_1d = np.clip(0.15 + estela * 0.5 + haz * 1.0, 0, 1)
    intensity_2d = np.tile(intensity_1d, (height, 1))

    # Contorno blanco animado
    stroke_alpha = (stroke_mask * intensity_2d * 255).clip(0, 255).astype(np.uint8)
    stroke_frame = np.zeros((height, width, 4), dtype=np.uint8)
    stroke_frame[:, :, :3] = 255
    stroke_frame[:, :, 3]  = stroke_alpha

    # Halo blanco difuso en el punto del haz
    haz_alpha = (stroke_mask * np.tile(haz, (height, 1)) * 200).clip(0,255).astype(np.uint8)
    halo_arr  = np.zeros((height, width, 4), dtype=np.uint8)
    halo_arr[:, :, :3] = 255
    halo_arr[:, :, 3]  = haz_alpha
    halo = Image.fromarray(halo_arr, "RGBA").filter(ImageFilter.GaussianBlur(radius=6))

    # Compositar: relleno neón → halo → contorno animado
    result = Image.fromarray(fill_arr, "RGBA")
    result = Image.alpha_composite(result, halo)
    result = Image.alpha_composite(result, Image.fromarray(stroke_frame, "RGBA"))
    return result


def generate_neon_overlay(
    text: str,
    width: int,
    height: int,
    fps: int,
    output_path: Path,
    config: dict,
) -> Path:
    from PIL import ImageFont

    layout      = config.get("layout", {})
    sweep_speed = layout.get("neon_sweep_speed", 0.6)
    font_name   = config.get("subtitles", {}).get("font_name", "Showcard Gothic")
    font_size   = max(12, int(height * layout.get("neon_font_size_ratio", 0.65)))
    bgr         = layout.get("webcam_name_color_bgr", [255, 50, 255])
    neon_rgb    = (int(bgr[2]), int(bgr[1]), int(bgr[0]))

    font_path = _find_font(font_name)
    if not font_path:
        logger.warning(f"Fuente '{font_name}' no encontrada, usando PIL default")
    try:
        font = ImageFont.truetype(font_path, font_size) if font_path else ImageFont.load_default()
    except Exception as e:
        logger.warning(f"Error cargando fuente ({e}), usando default")
        font = ImageFont.load_default()

    cycle_frames = max(10, int(round(fps / sweep_speed)))
    fill_arr, stroke_mask = _build_static_layers(text, width, height, font, neon_rgb)

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)

        def render_frame(i: int) -> None:
            t = i / cycle_frames
            frame = _render_frame(fill_arr, stroke_mask, width, height, t)
            frame.save(str(td / f"f{i:04d}.png"))

        workers = min(os.cpu_count() or 4, 8)
        with ThreadPoolExecutor(max_workers=workers) as executor:
            list(executor.map(render_frame, range(cycle_frames)))

        r = subprocess.run([
            "ffmpeg", "-y",
            "-framerate", str(fps),
            "-i", str(td / "f%04d.png"),
            "-vcodec", "png", "-pix_fmt", "rgba",
            str(output_path),
        ], capture_output=True, timeout=60)
        if r.returncode != 0:
            raise RuntimeError(
                f"Error generando overlay:\n"
                f"{r.stderr.decode(errors='ignore')[-400:]}"
            )

    logger.info(f"Overlay trazo: {output_path.name} ({cycle_frames} frames, {width}x{height}px)")
    return output_path
