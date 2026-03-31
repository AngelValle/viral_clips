"""
dynamic_crop.py
Encuadre dinámico automático: sigue el elemento de acción frame a frame.

Algoritmo:
  1. Diferencia entre frames consecutivos → mapa de movimiento
  2. Centroide del movimiento → target (cx, cy)
  3. Zoom dinámico: si el movimiento está concentrado en un área pequeña
     (conversación, cara hablando) → zoom in hasta zoom_max
  4. Suavizado exponencial separado para posición y zoom → sin saltos bruscos
  5. Output: vídeo recortado y escalado a res_w x res_h

El resultado se entrega a composer.py como si fuera el raw_clip,
pero ya con el encuadre correcto aplicado.

Parámetros configurables en config.json bajo "dynamic_crop":
  enabled           : true/false (default: true)
  zoom_base         : zoom mínimo del crop (default: valor de layout.gameplay_zoom)
  zoom_max          : zoom máximo en conversación/cara (default: zoom_base * 1.4)
  motion_alpha      : suavizado de posición 0.0–1.0 (default: 0.08, más alto = más rápido)
  zoom_alpha        : suavizado de zoom 0.0–1.0 (default: 0.03, más lento que posición)
  motion_threshold  : umbral diferencia entre frames para contar movimiento (default: 20)
  concentration_max : área máxima de movimiento (ratio 0–1) para activar zoom (default: 0.15)
  hud_exclude_ratio : fracción inferior/lateral a excluir del análisis (HUD) (default: 0.15)
  x_offset          : offset horizontal fijo sobre el centroide (default: 0)
"""

import logging
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def _smooth(current: float, target: float, alpha: float) -> float:
    """Suavizado exponencial. alpha=1 → instantáneo, alpha=0 → sin movimiento."""
    return current + alpha * (target - current)


def compute_tracking(
    clip_path: Path,
    config: dict,
    out_w: int,
    out_h: int,
) -> list:
    """
    Analiza el clip y devuelve una lista de dicts por frame:
      {"cx": int, "cy": int, "zoom": float}
    donde cx, cy son el centro del crop en coordenadas de la fuente.
    """
    dc      = config.get("dynamic_crop", {})
    layout  = config.get("layout", {})

    motion_alpha  = dc.get("motion_alpha",      0.03)
    zoom_alpha    = dc.get("zoom_alpha",         0.01)
    motion_thresh = dc.get("motion_threshold",   25)
    conc_max      = dc.get("concentration_max",  0.10)
    hud_excl      = dc.get("hud_exclude_ratio",  0.15)
    x_offset      = dc.get("x_offset",           0)
    skip_y        = layout.get("stats_skip_ratio", 0.0)

    cap    = cv2.VideoCapture(str(clip_path))
    src_w  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # zoom_base: mínimo para llenar out_w x out_h sin bandas negras.
    # Si el usuario lo configura explícitamente se respeta, si no se calcula.
    if "zoom_base" in dc:
        zoom_base = float(dc["zoom_base"])
    else:
        # Zoom mínimo para que el crop en aspect 9:16 cubra el frame fuente
        zoom_base = max(src_w / src_h * out_h / out_w, 1.0)

    # zoom_max: máximo zoom-in para conversaciones. Muy conservador (+15% por defecto).
    zoom_max = dc.get("zoom_max", round(zoom_base * 1.15, 3))

    # Zona del gameplay excluyendo HUD (inferior y lados)
    analysis_y1 = int(src_h * skip_y)
    analysis_y2 = int(src_h * (1.0 - hud_excl))
    analysis_x1 = int(src_w * hud_excl * 0.5)
    analysis_x2 = int(src_w * (1.0 - hud_excl * 0.5))

    # Estado inicial: centrado
    cx   = src_w / 2 + x_offset
    cy   = (analysis_y1 + analysis_y2) / 2
    zoom = zoom_base

    prev_gray = None
    keyframes = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        roi     = gray[analysis_y1:analysis_y2, analysis_x1:analysis_x2]
        roi_h, roi_w = roi.shape

        target_cx   = cx
        target_cy   = cy
        target_zoom = zoom_base  # por defecto volver al zoom base

        if prev_gray is not None:
            prev_roi = prev_gray[analysis_y1:analysis_y2, analysis_x1:analysis_x2]

            # Mapa de movimiento
            diff = cv2.absdiff(roi, prev_roi)
            motion_mask = (diff > motion_thresh).astype(np.uint8)

            motion_px = motion_mask.sum()
            total_px  = roi_h * roi_w

            if motion_px > 50:  # mínimo de movimiento significativo
                # Centroide del movimiento
                ys, xs = np.where(motion_mask > 0)
                # Volver a coordenadas de fuente
                target_cx = float(xs.mean()) + analysis_x1 + x_offset
                target_cy = float(ys.mean()) + analysis_y1

                # Ratio de concentración: movimiento en área pequeña → zoom in
                motion_ratio = motion_px / total_px
                if motion_ratio < conc_max:
                    # Movimiento muy concentrado (cara hablando, conversación)
                    # Zoom sutil: máximo +15% sobre zoom_base
                    t_factor    = 1.0 - (motion_ratio / conc_max)  # 0→1
                    target_zoom = zoom_base + t_factor * (zoom_max - zoom_base)
                else:
                    # Movimiento amplio → mantener zoom base
                    target_zoom = zoom_base

        # Suavizado exponencial
        cx   = _smooth(cx,   target_cx,   motion_alpha)
        cy   = _smooth(cy,   target_cy,   motion_alpha)
        zoom = _smooth(zoom, target_zoom, zoom_alpha)

        # Calcular crop en coordenadas de fuente
        crop_w = int(src_w / zoom)
        crop_h = int(src_h / zoom)

        # Asegurar que el crop mantiene el aspect ratio de salida
        out_aspect = out_w / out_h
        if crop_w / crop_h > out_aspect:
            crop_w = int(crop_h * out_aspect)
        else:
            crop_h = int(crop_w / out_aspect)

        # Crop centrado en (cx, cy), clampeado a los bordes
        x1 = int(cx - crop_w / 2)
        y1 = int(cy - crop_h / 2)
        x1 = max(0, min(x1, src_w - crop_w))
        y1 = max(0, min(y1, src_h - crop_h))

        keyframes.append({"x": x1, "y": y1, "w": crop_w, "h": crop_h})
        prev_gray = gray

    cap.release()

    # Si no hay keyframes (clip vacío), devolver crop estático centrado
    if not keyframes:
        crop_w = int(src_w / zoom_base)
        crop_h = int(src_h / zoom_base)
        cx_s   = int(src_w / 2 + x_offset - crop_w / 2)
        cy_s   = int(src_h / 2 - crop_h / 2)
        keyframes = [{"x": cx_s, "y": cy_s, "w": crop_w, "h": crop_h}]

    logger.info(
        f"Tracking: {len(keyframes)} frames | "
        f"zoom {zoom_base:.2f}x–{zoom_max:.2f}x | "
        f"X rango [{min(k['x'] for k in keyframes)}–{max(k['x'] for k in keyframes)}] "
        f"Y rango [{min(k['y'] for k in keyframes)}–{max(k['y'] for k in keyframes)}]"
    )
    return keyframes


def apply_dynamic_crop(
    clip_path: Path,
    keyframes: list,
    out_w: int,
    out_h: int,
    fps: float,
    output_path: Path,
    hw: dict,
) -> Path:
    """
    Aplica el crop dinámico frame a frame via pipe Python → FFmpeg NVENC.
    Preserva el audio del clip original.
    """
    if not keyframes:
        raise ValueError("keyframes vacío")

    cap = cv2.VideoCapture(str(clip_path))

    # Writer pipe hacia FFmpeg
    cmd_enc = [
        "ffmpeg", "-y",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-s", f"{out_w}x{out_h}",
        "-pix_fmt", "bgr24",
        "-r", str(fps),
        "-i", "pipe:0",
        # Audio del original
        "-i", str(clip_path),
        "-c:v", hw.get("encoder", "libx264"),
        *hw.get("extra_enc_args", ["-preset", "fast", "-crf", "18"]),
        "-c:a", "copy",
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-shortest",
        "-movflags", "+faststart",
        str(output_path),
    ]

    proc = subprocess.Popen(cmd_enc, stdin=subprocess.PIPE,
                            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

    frame_idx = 0
    n_kf      = len(keyframes)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        kf = keyframes[min(frame_idx, n_kf - 1)]
        x, y, w, h = kf["x"], kf["y"], kf["w"], kf["h"]

        # Recortar y escalar al tamaño de salida
        cropped = frame[y:y+h, x:x+w]
        if cropped.shape[0] == 0 or cropped.shape[1] == 0:
            # Fallback: frame negro
            scaled = np.zeros((out_h, out_w, 3), dtype=np.uint8)
        else:
            scaled = cv2.resize(cropped, (out_w, out_h), interpolation=cv2.INTER_LINEAR)

        proc.stdin.write(scaled.tobytes())
        frame_idx += 1

    cap.release()
    proc.stdin.close()
    _, stderr = proc.communicate()

    if proc.returncode != 0:
        err = stderr.decode(errors="ignore")
        raise RuntimeError(f"Error en dynamic crop encode:\n{err[-400:]}")

    logger.info(f"Dynamic crop aplicado: {output_path.name} ({frame_idx} frames)")
    return output_path
