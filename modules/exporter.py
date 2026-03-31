"""
exporter.py
Renderizado final del clip:
  1. Extrae el segmento crudo del vídeo de entrada.
  2. Embebe los subtítulos ASS en el vídeo.
  3. Garantiza la resolución, fps, codec y bitrate finales.
  4. Guarda el JSON de metadatos del clip.
Usa encoder hardware (NVENC/VideoToolbox/AMF) cuando está disponible.
"""

import json
import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Any

from modules.gpu_utils import get_ffmpeg_hwaccel

logger = logging.getLogger(__name__)


def extract_segment(
    source_path: Path,
    start: float,
    end: float,
    output_path: Path,
    fragments: list = None,
) -> Path:
    """
    Extrae uno o varios fragmentos del vídeo fuente y los concatena en un único clip.

    Si 'fragments' es None o tiene un único elemento, hace un trim simple con
    -accurate_seek (rápido, frame-exact).

    Si 'fragments' tiene varios elementos, usa filter_complex trim+concat para
    ensamblar los fragmentos con jump cuts, recodificando con NVENC/libx264.
    El audio se recodifica a AAC en ambos casos para garantizar sincronía.
    """
    from modules.gpu_utils import get_ffmpeg_hwaccel
    hw = get_ffmpeg_hwaccel()

    # Normalizar: si no hay fragments usar start/end como fragmento único
    if not fragments:
        fragments = [{"start": start, "end": end}]

    # ── Caso simple: un único fragmento ──────────────────────────────────────
    if len(fragments) == 1:
        fs = fragments[0]["start"]
        fe = fragments[0]["end"]
        duration = round(fe - fs, 3)

        def build_simple(encoder, extra_args):
            return [
                "ffmpeg", "-y",
                "-accurate_seek",
                "-ss", str(fs),
                "-i", str(source_path),
                "-t", str(duration),
                "-c:v", encoder, *extra_args,
                "-c:a", "aac", "-b:a", "192k",
                "-avoid_negative_ts", "make_zero",
                str(output_path),
            ]

        result = subprocess.run(
            build_simple(hw["encoder"], hw["extra_enc_args"]),
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE
        )
        if result.returncode != 0 and hw["is_hw"]:
            result = subprocess.run(
                build_simple("libx264", ["-preset", "fast", "-crf", "18"]),
                stdout=subprocess.DEVNULL, stderr=subprocess.PIPE
            )
        if result.returncode != 0:
            logger.error(f"Error extrayendo segmento: {result.stderr.decode(errors='ignore')}")
            raise RuntimeError(f"Fallo al extraer segmento [{fs:.2f}-{fe:.2f}]")
        logger.debug(f"Segmento extraído (1 fragmento): {output_path.name}")
        return output_path

    # ── Caso múltiple: varios fragmentos con jump cuts ────────────────────────
    n = len(fragments)
    filter_parts = []
    v_labels = []
    a_labels = []

    for i, frag in enumerate(fragments):
        fs  = round(frag["start"], 3)
        dur = round(frag["end"] - frag["start"], 3)
        filter_parts.append(
            f"[0:v]trim=start={fs}:duration={dur},setpts=PTS-STARTPTS[v{i}]"
        )
        filter_parts.append(
            f"[0:a]atrim=start={fs}:duration={dur},asetpts=PTS-STARTPTS[a{i}]"
        )
        v_labels.append(f"[v{i}]")
        a_labels.append(f"[a{i}]")

    filter_parts.append("".join(v_labels) + f"concat=n={n}:v=1:a=0[vout]")
    filter_parts.append("".join(a_labels) + f"concat=n={n}:v=0:a=1[aout]")
    filter_complex = ";".join(filter_parts)

    def build_multi(encoder, extra_args):
        return [
            "ffmpeg", "-y",
            "-i", str(source_path),
            "-filter_complex", filter_complex,
            "-map", "[vout]", "-map", "[aout]",
            "-c:v", encoder, *extra_args,
            "-c:a", "aac", "-b:a", "192k",
            "-avoid_negative_ts", "make_zero",
            "-movflags", "+faststart",
            str(output_path),
        ]

    result = subprocess.run(
        build_multi(hw["encoder"], hw["extra_enc_args"]),
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE
    )
    if result.returncode != 0 and hw["is_hw"]:
        err = result.stderr.decode(errors="ignore")
        logger.warning(f"NVENC falló en extracción multi-fragmento, reintentando con libx264:\n{err[-300:]}")
        result = subprocess.run(
            build_multi("libx264", ["-preset", "fast", "-crf", "18"]),
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE
        )
    if result.returncode != 0:
        logger.error(f"Error extrayendo multi-fragmento: {result.stderr.decode(errors='ignore')[-400:]}")
        raise RuntimeError(f"Fallo al extraer {n} fragmentos")

    total = sum(f["end"] - f["start"] for f in fragments)
    logger.debug(f"Segmento extraído ({n} fragmentos, {total:.1f}s total): {output_path.name}")
    return output_path


def _safe_ass_path(ass_path: Path) -> tuple:
    """
    Copia el fichero .ass a una ruta temporal sin espacios ni caracteres especiales.
    Devuelve (ruta_segura, necesita_limpiar).
    En Windows, rutas con espacios o caracteres Unicode rompen el filtro ass= de ffmpeg.
    """
    ass_str = str(ass_path)
    # Comprobar si la ruta tiene caracteres problemáticos para ffmpeg
    problematic = any(c in ass_str for c in (' ', '(', ')', '[', ']', ',', ';', "'", '"', '@'))

    if problematic:
        # Copiar a carpeta temporal con nombre simple
        tmp_dir  = Path(tempfile.gettempdir()) / "vc_subs"
        tmp_dir.mkdir(exist_ok=True)
        safe_path = tmp_dir / "subtitle.ass"
        shutil.copy2(str(ass_path), str(safe_path))
        logger.debug(f"ASS copiado a ruta segura: {safe_path}")
        return safe_path, True

    return ass_path, False


def _build_ass_vf(ass_path: Path, res_w: int = 1080, res_h: int = 1920) -> str:
    """
    Construye el filtro -vf para embeber subtítulos ASS compatible con FFmpeg 8.x.
    FFmpeg 8 requiere original_size=WxH para el filtro ass=.
    """
    import re
    path_str = str(ass_path).replace("\\", "/")
    # Escapar letra de unidad Windows: C:/ → C\:/
    path_str = re.sub(r"^([A-Za-z]):", r"\1\\:", path_str)
    return f"ass='{path_str}':original_size={res_w}x{res_h}"

def embed_subtitles(
    video_path: Path,
    ass_path: Path,
    output_path: Path,
    config: dict,
) -> Path:
    """
    Embebe los subtítulos ASS en el vídeo con ffmpeg (hard subtitles).
    Maneja correctamente rutas de Windows con espacios y caracteres especiales.
    Usa encoder hardware cuando está disponible, libx264 como fallback.
    """
    cfg_o = config["output"]
    res_w = cfg_o["resolution_w"]
    res_h = cfg_o["resolution_h"]
    fps   = cfg_o["fps"]
    hw    = get_ffmpeg_hwaccel()

    # Asegurar ruta sin caracteres problemáticos
    safe_ass, cleanup = _safe_ass_path(ass_path)
    vf = _build_ass_vf(safe_ass, res_w=res_w, res_h=res_h)

    logger.info(f"Render final con encoder: {hw['encoder']} ({'GPU' if hw['is_hw'] else 'CPU'})")

    def build_cmd(encoder: str, extra_args: list, hwaccel: str = None) -> list:
        cmd = ["ffmpeg", "-y"]
        if hwaccel == "cuda":
            cmd += ["-hwaccel", "cuda"]
        elif hwaccel == "videotoolbox":
            cmd += ["-hwaccel", "videotoolbox"]
        cmd += [
            "-i", str(video_path),
            "-vf", vf,
            "-s", f"{res_w}x{res_h}",
            "-r", str(fps),
            "-c:v", encoder,
            *extra_args,
            "-c:a", cfg_o["audio_codec"],
            "-b:a", "192k",
            "-movflags", "+faststart",
            str(output_path),
        ]
        return cmd

    logger.info("Embebiendo subtítulos y renderizando clip final...")

    # Intento 1: encoder hardware
    result = subprocess.run(
        build_cmd(hw["encoder"], hw["extra_enc_args"], hw["hwaccel"]),
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE
    )

    # Intento 2: fallback libx264
    if result.returncode != 0:
        err = result.stderr.decode(errors="ignore")
        logger.warning(f"Encoder {hw['encoder']} falló, reintentando con libx264...\n{err[:200]}")
        result = subprocess.run(
            build_cmd("libx264", ["-preset", "fast", "-crf", "23"]),
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE
        )

    # Limpiar ASS temporal si se creó
    if cleanup:
        try:
            os.unlink(str(safe_ass))
        except Exception:
            pass

    if result.returncode != 0:
        logger.error(f"Error embed subs: {result.stderr.decode(errors='ignore')}")
        raise RuntimeError("Fallo al embeber subtítulos")

    logger.info(f"Clip final renderizado: {output_path.name}")
    return output_path


def save_metadata(
    output_path: Path,
    source_path: Path,
    segment: Dict[str, Any],
    clip_words: list,
    config: dict,
    metadata_dir: Path,
) -> Path:
    """Guarda un JSON con metadatos del clip."""
    meta = {
        "clip_file":      output_path.name,
        "source_file":    source_path.name,
        "start_sec":      segment["start"],
        "end_sec":        segment["end"],
        "duration_sec":   round(segment["end"] - segment["start"], 2),
        "viral_score":    segment.get("score", 0),
        "resolution":     f"{config['output']['resolution_w']}x{config['output']['resolution_h']}",
        "fps":            config["output"]["fps"],
        "word_count":     len(clip_words),
        "censored_words": [w["word"] for w in clip_words if w.get("censored")],
        "transcript":     " ".join(w.get("censored_text", w["word"]) for w in clip_words),
    }
    meta_path = metadata_dir / (output_path.stem + ".json")
    meta_path.write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info(f"Metadatos guardados: {meta_path.name}")
    return meta_path


def remap_words_to_fragments(words: list, fragments: list) -> list:
    """
    Reasigna los timestamps de las palabras al nuevo timeline generado
    por la concatenación de fragmentos con jump cuts.

    Ejemplo con 2 fragmentos [10s-15s] y [30s-35s]:
      - Fragmento 0: ocupa t=0s a t=5s en el clip final
      - Fragmento 1: ocupa t=5s a t=10s en el clip final
      - Palabra en t=31s del directo → t=6s en el clip final

    Las palabras que caen en los gaps entre fragmentos (silencios cortados)
    se descartan silenciosamente.
    """
    # Construir mapa: (orig_start, orig_end) → new_start en el clip
    cursor = 0.0
    frag_map = []
    for frag in sorted(fragments, key=lambda f: f["start"]):
        fs = frag["start"]
        fe = frag["end"]
        frag_map.append({"orig_start": fs, "orig_end": fe, "new_start": cursor})
        cursor += fe - fs

    remapped = []
    for w in words:
        new_start = None
        new_end   = None

        for seg in frag_map:
            # Palabra completamente dentro del fragmento
            if seg["orig_start"] <= w["start"] and w["end"] <= seg["orig_end"]:
                new_start = seg["new_start"] + (w["start"] - seg["orig_start"])
                new_end   = seg["new_start"] + (w["end"]   - seg["orig_start"])
                break
            # Palabra que solapa el inicio del fragmento (padding)
            if w["start"] < seg["orig_start"] < w["end"]:
                new_start = seg["new_start"]
                new_end   = seg["new_start"] + (w["end"] - seg["orig_start"])
                break
            # Palabra que solapa el final del fragmento (padding)
            if w["start"] < seg["orig_end"] < w["end"]:
                new_start = seg["new_start"] + (w["start"] - seg["orig_start"])
                new_end   = seg["new_start"] + (seg["orig_end"] - seg["orig_start"])
                break

        if new_start is None:
            continue  # Palabra en gap cortado — descartar

        remapped.append({**w,
                         "start": round(max(0.0, new_start), 3),
                         "end":   round(max(0.0, new_end),   3)})
    return remapped
