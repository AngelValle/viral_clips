"""
tools/measure_border_ratio.py

Mide el border_min_ratio real del borde morado de la webcam
en un frame concreto del vídeo.

Uso:
    python tools/measure_border_ratio.py --file videos/input/video.mp4
    python tools/measure_border_ratio.py --file videos/input/video.mp4 --second 120
    python tools/measure_border_ratio.py --file videos/input/video.mp4 --second 120 --second2 300

Muestra una ventana con el borde analizado y el ratio exacto.
Así sabes exactamente qué valor poner en border_min_ratio.
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


def extract_frame(video_path: str, second: int) -> str:
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    tmp.close()
    subprocess.run([
        "ffmpeg", "-y", "-ss", str(second), "-i", video_path,
        "-vframes", "1", "-q:v", "2", tmp.name,
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return tmp.name


def measure_ratio(frame, cam_x, cam_y, cam_w, cam_h,
                  hsv_lower, hsv_upper, border_px) -> tuple:
    """
    Devuelve (ratio, purple_mask, border_mask) para visualización.
    """
    import cv2
    import numpy as np

    h, w = frame.shape[:2]
    x1 = max(0, cam_x - border_px)
    y1 = max(0, cam_y - border_px)
    x2 = min(w, cam_x + cam_w + border_px)
    y2 = min(h, cam_y + cam_h + border_px)

    region = frame[y1:y2, x1:x2]

    inner_x1 = border_px
    inner_y1 = border_px
    inner_x2 = region.shape[1] - border_px
    inner_y2 = region.shape[0] - border_px

    import numpy as np
    mask = np.ones(region.shape[:2], dtype=np.uint8) * 255
    if inner_x2 > inner_x1 and inner_y2 > inner_y1:
        mask[inner_y1:inner_y2, inner_x1:inner_x2] = 0

    hsv    = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    lower  = np.array(hsv_lower, dtype=np.uint8)
    upper  = np.array(hsv_upper, dtype=np.uint8)
    purple = cv2.inRange(hsv, lower, upper)
    purple_border = cv2.bitwise_and(purple, mask)

    border_pixels = int((mask > 0).sum())
    purple_pixels = int((purple_border > 0).sum())
    ratio = purple_pixels / border_pixels if border_pixels > 0 else 0.0

    return ratio, purple_border, mask, region, (x1, y1, x2, y2)


def analyze_frame(frame, config: dict, second: int, label: str = ""):
    import cv2
    import numpy as np

    layout = config.get("layout", {})
    cfg_f  = config.get("face_detection", {})

    src_h, src_w = frame.shape[:2]

    w_ratio  = layout.get("webcam_w_ratio", 0.137)
    h_ratio  = layout.get("webcam_h_ratio", 0.331)
    x_offset = layout.get("webcam_x_offset", 22)
    y_center = layout.get("webcam_y_center_ratio", 0.5)

    cam_w  = int(src_w * w_ratio)
    cam_h  = int(src_h * h_ratio)
    cam_x  = x_offset
    cam_y  = int(src_h * y_center) - cam_h // 2
    cam_y  = max(0, min(cam_y, src_h - cam_h))

    hsv_lower  = cfg_f.get("border_color_hsv_lower", [125, 50, 50])
    hsv_upper  = cfg_f.get("border_color_hsv_upper", [165, 255, 255])
    border_px  = cfg_f.get("border_px", 8)
    min_ratio  = cfg_f.get("border_min_ratio", 0.03)

    ratio, purple_border, border_mask, region, (x1, y1, x2, y2) = \
        measure_ratio(frame, cam_x, cam_y, cam_w, cam_h,
                      hsv_lower, hsv_upper, border_px)

    detected = ratio >= min_ratio

    print()
    print(f"{'='*55}")
    print(f"Segundo {second}s  {label}")
    print(f"{'='*55}")
    print(f"  Zona webcam:       ({cam_x},{cam_y}) {cam_w}x{cam_h}px")
    print(f"  HSV lower/upper:   {hsv_lower} / {hsv_upper}")
    print(f"  border_px:         {border_px}")
    print()
    print(f"  >> border_min_ratio MEDIDO: {ratio:.4f}  ({ratio*100:.1f}%)")
    print()
    print(f"  border_min_ratio actual:    {min_ratio}")
    print(f"  Cámara detectada:           {'✓ SÍ' if detected else '✗ NO'}")
    print()

    if not detected and ratio > 0:
        suggested = round(ratio * 0.8, 3)
        print(f"  → Para detectar este frame: border_min_ratio <= {suggested}")
    elif detected:
        suggested = round(ratio * 1.2, 3)
        print(f"  → Para rechazar este frame: border_min_ratio >= {suggested}")
    print(f"{'='*55}")

    # Visualización
    vis = frame.copy()

    # Rectángulo zona webcam (verde)
    cv2.rectangle(vis, (cam_x, cam_y), (cam_x + cam_w, cam_y + cam_h),
                  (0, 255, 0), 3)

    # Rectángulo zona borde analizada (azul)
    cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 100, 0), 2)

    # Píxeles morados detectados (overlay magenta)
    purple_vis = np.zeros_like(frame)
    purple_vis[y1:y2, x1:x2][purple_border > 0] = [255, 0, 255]
    vis = cv2.addWeighted(vis, 0.8, purple_vis, 0.8, 0)

    # Texto con ratio
    color = (0, 200, 0) if detected else (0, 0, 255)
    status = "DETECTADA" if detected else "NO detectada"
    cv2.putText(vis, f"s={second}  ratio={ratio:.3f}  [{status}]",
                (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
    cv2.putText(vis, f"border_min_ratio actual: {min_ratio}",
                (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    return vis, ratio


def main():
    parser = argparse.ArgumentParser(
        description="Mide el border_min_ratio real del borde morado en frames concretos."
    )
    parser.add_argument("--file",    required=True, help="Ruta al vídeo")
    parser.add_argument("--second",  type=int, default=60,
                        help="Primer segundo a analizar (default: 60)")
    parser.add_argument("--second2", type=int, default=None,
                        help="Segundo adicional a analizar (opcional)")
    parser.add_argument("--config",  type=str, default="config.json",
                        help="Ruta al config.json (default: config.json)")
    args = parser.parse_args()

    try:
        import cv2
        import numpy as np
    except ImportError:
        print("ERROR: opencv-python no instalado.")
        sys.exit(1)

    config = json.loads(Path(args.config).read_text())

    seconds = [args.second]
    if args.second2 is not None:
        seconds.append(args.second2)

    frames_vis = []
    for sec in seconds:
        print(f"Extrayendo frame en segundo {sec}s...")
        path = extract_frame(args.file, sec)
        frame = cv2.imread(path)
        os.unlink(path)
        if frame is None:
            print(f"ERROR: No se pudo extraer frame en segundo {sec}")
            continue
        label = "(con cámara?)" if sec == seconds[0] else "(sin cámara?)"
        vis, ratio = analyze_frame(frame, config, sec, label)
        frames_vis.append((sec, vis, ratio))

    print()
    print("Abriendo ventana de visualización...")
    print("  Rosa/magenta = píxeles morados detectados")
    print("  Verde        = zona webcam configurada")
    print("  Azul         = zona de borde analizada")
    print("  Pulsa cualquier tecla para cerrar")

    max_w = 900
    for sec, vis, ratio in frames_vis:
        h, w = vis.shape[:2]
        scale = min(1.0, max_w / w)
        small = cv2.resize(vis, (int(w * scale), int(h * scale)))
        cv2.imshow(f"Segundo {sec}s — ratio={ratio:.4f}", small)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
