"""
face_detector.py
Detección de presencia de cámara en dos niveles:

  Nivel 1 — Borde morado (rápido):
    Analiza los píxeles del contorno de la zona webcam buscando el color
    morado/púrpura característico del borde del stream.

  Nivel 2 — Cara (solo si nivel 1 pasa):
    Si el borde es morado pero podría ser iluminación ambiental,
    confirma que hay una cara dentro del recuadro usando el detector
    Haar de OpenCV (incluido en opencv-python, sin dependencias extra).

Si el color del borde cambia, ajusta en config.json:
  face_detection.border_color_hsv_lower: [H_min, S_min, V_min]
  face_detection.border_color_hsv_upper: [H_max, S_max, V_max]
  (valores en espacio HSV de OpenCV: H=0-179, S=0-255, V=0-255)

Para desactivar el nivel 2: face_detection.require_face: false
"""

import logging
import numpy as np
import cv2
from pathlib import Path
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

DEFAULT_HSV_LOWER        = [125, 50,  50]
DEFAULT_HSV_UPPER        = [165, 255, 255]
DEFAULT_BORDER_MIN_RATIO = 0.03


# ── Nivel 1: borde morado ─────────────────────────────────────────────────────

def _has_purple_border(frame: np.ndarray,
                       cam_x: int, cam_y: int, cam_w: int, cam_h: int,
                       hsv_lower: list, hsv_upper: list,
                       min_ratio: float,
                       border_px: int = 8) -> bool:
    h, w = frame.shape[:2]

    x1 = max(0, cam_x - border_px)
    y1 = max(0, cam_y - border_px)
    x2 = min(w, cam_x + cam_w + border_px)
    y2 = min(h, cam_y + cam_h + border_px)

    region = frame[y1:y2, x1:x2]
    if region.size == 0:
        return False

    inner_x1 = border_px
    inner_y1 = border_px
    inner_x2 = region.shape[1] - border_px
    inner_y2 = region.shape[0] - border_px

    mask = np.ones(region.shape[:2], dtype=np.uint8) * 255
    if inner_x2 > inner_x1 and inner_y2 > inner_y1:
        mask[inner_y1:inner_y2, inner_x1:inner_x2] = 0

    hsv    = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    lower  = np.array(hsv_lower, dtype=np.uint8)
    upper  = np.array(hsv_upper, dtype=np.uint8)
    purple = cv2.inRange(hsv, lower, upper)
    purple = cv2.bitwise_and(purple, mask)

    border_pixels = np.count_nonzero(mask)
    purple_pixels = np.count_nonzero(purple)
    ratio         = purple_pixels / border_pixels if border_pixels > 0 else 0.0

    return ratio >= min_ratio


# ── Nivel 2: cara dentro del recuadro ────────────────────────────────────────

_cascade: Optional[cv2.CascadeClassifier] = None

def _get_cascade() -> cv2.CascadeClassifier:
    global _cascade
    if _cascade is None:
        path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        _cascade = cv2.CascadeClassifier(path)
        if _cascade.empty():
            raise RuntimeError(f"No se pudo cargar el cascade Haar: {path}")
    return _cascade


def _has_face_inside(frame: np.ndarray,
                     cam_x: int, cam_y: int, cam_w: int, cam_h: int,
                     scale_factor: float = 1.1,
                     min_neighbors: int  = 3,
                     min_size_ratio: float = 0.15) -> bool:
    """
    Busca una cara dentro del recuadro de la webcam.
    min_size_ratio: tamaño mínimo de cara como fracción del ancho del recuadro.
    """
    h, w = frame.shape[:2]
    x1 = max(0, cam_x)
    y1 = max(0, cam_y)
    x2 = min(w, cam_x + cam_w)
    y2 = min(h, cam_y + cam_h)

    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return False

    gray     = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    min_size = max(20, int(cam_w * min_size_ratio))

    cascade = _get_cascade()
    faces   = cascade.detectMultiScale(
        gray,
        scaleFactor  = scale_factor,
        minNeighbors = min_neighbors,
        minSize      = (min_size, min_size),
        flags        = cv2.CASCADE_SCALE_IMAGE,
    )
    return len(faces) > 0


# ── Suavizado de transiciones ─────────────────────────────────────────────────

def _smooth_face_data(face_data: List[Dict], buffer_frames: int = 30) -> List[Dict]:
    """Suaviza transiciones para evitar parpadeos."""
    n        = len(face_data)
    smoothed = [d.copy() for d in face_data]

    last_face = -1
    for i in range(n):
        if face_data[i]["has_face"]:
            last_face = i
        elif last_face >= 0 and (i - last_face) <= buffer_frames:
            smoothed[i]["has_face"] = True
            smoothed[i]["bbox"]     = face_data[last_face]["bbox"]

    return smoothed




# ── Detección de modo conducción ─────────────────────────────────────────────

def _is_driving_frame(frame: np.ndarray, config: dict) -> bool:
    """
    Detecta si el frame es de conducción buscando los textos fijos del HUD
    del velocímetro: "HBK", "ABS", "GEAR", "KMH".

    Estos textos son letras blancas pequeñas sobre fondo oscuro, siempre
    presentes cuando se conduce. La detección usa componentes conectados
    de píxeles blancos en la ROI — sin OCR, sin modelos, ~0.1ms por frame.

    Con velocímetro:  ~12 componentes blancos (4 palabras × ~3 letras)
    Sin velocímetro:  <5  componentes blancos (gameplay variado)

    Configurable en config.json bajo layout:
      driving_hud_min_components : mínimo de componentes para detectar HUD (default: 6)
      driving_hud_white_thresh   : umbral de brillo para "blanco" (default: 200)
      driving_hud_min_area       : área mínima de componente válido en px (default: 20)
      driving_hud_max_area       : área máxima de componente válido en px (default: 800)
    """
    layout     = config.get("layout", {})
    min_comps  = layout.get("driving_hud_min_components", 6)
    w_thresh   = layout.get("driving_hud_white_thresh",   200)
    min_area   = layout.get("driving_hud_min_area",       20)
    max_area   = layout.get("driving_hud_max_area",       800)

    h, w = frame.shape[:2]
    # ROI calibrada sobre captura real: esquina inf-der donde aparecen HBK/ABS/GEAR/KMH
    roi = frame[int(h * 0.88):int(h * 0.99),
                int(w * 0.87):int(w * 0.99)]
    if roi.size == 0:
        return False

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, w_thresh, 255, cv2.THRESH_BINARY)

    n_labels, _, stats, _ = cv2.connectedComponentsWithStats(thresh)

    # Contar componentes del tamaño de una letra del HUD
    valid_components = sum(
        1 for i in range(1, n_labels)
        if min_area < stats[i, cv2.CC_STAT_AREA] < max_area
    )
    return valid_components >= min_comps


# ── Punto de entrada ──────────────────────────────────────────────────────────

def analyze_video_faces(clip_path: Path, config: dict) -> List[Dict[str, Any]]:
    """
    Analiza cada frame del clip con detección en dos niveles:
      1. Borde morado → rápido, procesa 1 de cada 2 frames
      2. Cara dentro del recuadro → solo cuando el borde pasa el umbral

    Si ambos niveles pasan → cámara confirmada.
    Si solo pasa el borde → posible falso positivo por iluminación → descartado.
    """
    cfg_f        = config["face_detection"]
    layout       = config.get("layout", {})
    buf_frames   = cfg_f.get("transition_buffer_frames", 30)
    require_face = cfg_f.get("require_face", True)

    cap     = cv2.VideoCapture(str(clip_path))
    src_w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    cam_w  = int(src_w * layout.get("webcam_w_ratio",       0.137))
    cam_h  = int(src_h * layout.get("webcam_h_ratio",       0.331))
    cam_x  = layout.get("webcam_x_offset", 22)
    cam_y  = int(src_h * layout.get("webcam_y_center_ratio", 0.5)) - cam_h // 2
    cam_y  = max(0, min(cam_y, src_h - cam_h))

    hsv_lower  = cfg_f.get("border_color_hsv_lower", DEFAULT_HSV_LOWER)
    hsv_upper  = cfg_f.get("border_color_hsv_upper", DEFAULT_HSV_UPPER)
    min_ratio  = cfg_f.get("border_min_ratio",        DEFAULT_BORDER_MIN_RATIO)
    border_px  = cfg_f.get("border_px",               8)

    # Parámetros del detector facial (ajustables en config)
    scale_factor    = cfg_f.get("face_scale_factor",    1.1)
    min_neighbors   = cfg_f.get("face_min_neighbors",   3)
    min_size_ratio  = cfg_f.get("face_min_size_ratio",  0.15)

    logger.info(
        f"Detección de cámara — nivel 1: borde morado | "
        f"nivel 2: cara {'activado' if require_face else 'desactivado'} | "
        f"zona: ({cam_x},{cam_y}) {cam_w}x{cam_h}px | frames: {total_f}"
    )

    video_fps      = cap.get(cv2.CAP_PROP_FPS) or 60.0
    face_data      = []
    frame_idx      = 0
    sample_every   = 2
    border_hits    = 0
    face_hits      = 0
    false_pos      = 0
    driving_hits   = 0
    driving_checks = 0
    driving_every  = max(1, int(video_fps * 2))  # cada ~2s

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % sample_every == 0:
            if frame_idx % driving_every == 0:
                driving_checks += 1
                if _is_driving_frame(frame, config):
                    driving_hits += 1

            has_border = _has_purple_border(
                frame, cam_x, cam_y, cam_w, cam_h,
                hsv_lower, hsv_upper, min_ratio, border_px
            )

            if has_border:
                border_hits += 1
                if require_face:
                    # Nivel 2: confirmar con cara
                    has_cam = _has_face_inside(
                        frame, cam_x, cam_y, cam_w, cam_h,
                        scale_factor, min_neighbors, min_size_ratio
                    )
                    if not has_cam:
                        false_pos += 1
                else:
                    has_cam = True

                if has_cam:
                    face_hits += 1
            else:
                has_cam = False

            face_data.append({
                "frame_idx":  frame_idx,
                "has_face":   has_cam,
                "bbox":       (cam_x, cam_y, cam_w, cam_h) if has_cam else None,
                "confidence": 1.0 if has_cam else 0.0,
            })
        else:
            prev = face_data[-1] if face_data else {"has_face": False, "bbox": None, "confidence": 0.0}
            face_data.append({
                "frame_idx":  frame_idx,
                "has_face":   prev["has_face"],
                "bbox":       prev["bbox"],
                "confidence": prev["confidence"],
            })

        frame_idx += 1

    cap.release()

    if cfg_f.get("smooth_tracking", True):
        face_data = _smooth_face_data(face_data, buffer_frames=buf_frames)

    cam_frames = sum(1 for d in face_data if d["has_face"])
    is_driving = driving_checks > 0 and (driving_hits / driving_checks) >= 0.5
    logger.info(
        f"Resultado — borde detectado: {border_hits} frames | "
        f"cara confirmada: {face_hits} frames | "
        f"falsos positivos descartados: {false_pos} | "
        f"cámara final: {cam_frames}/{len(face_data)} "
        f"({cam_frames/max(len(face_data),1)*100:.1f}%) | "
        f"conducción: {'SÍ' if is_driving else 'NO'} "
        f"({driving_hits}/{driving_checks} muestras)"
    )
    return face_data, is_driving
