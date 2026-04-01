#!/usr/bin/env python3
"""
tools/calibrate.py
Herramientas de calibración del pipeline viral_clips.

Subcomandos:
    gpu     — Diagnóstico completo de GPU (NVIDIA, PyTorch, FFmpeg NVENC)
    webcam  — Selección interactiva de la zona webcam en un frame del vídeo
    border  — Medición del border_min_ratio del borde morado de la webcam

Ejemplos:
    python tools/calibrate.py gpu
    python tools/calibrate.py webcam --file videos/input/video.mp4
    python tools/calibrate.py border --file videos/input/video.mp4 --second 120
    python tools/calibrate.py border --file videos/input/video.mp4 --second 60 --second2 300
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


# ── Shared helper ─────────────────────────────────────────────────────────────

def _extract_frame(video_path: str, second: int) -> str:
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    tmp.close()
    subprocess.run([
        "ffmpeg", "-y", "-ss", str(second), "-i", video_path,
        "-vframes", "1", "-q:v", "2", tmp.name,
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return tmp.name


# ── gpu ───────────────────────────────────────────────────────────────────────

def cmd_gpu(_args) -> None:
    SEP = "=" * 55

    # 1. Drivers NVIDIA
    print(f"\n{SEP}\n1. DRIVERS NVIDIA (nvidia-smi)\n{SEP}")
    try:
        r = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=20)
        if r.returncode == 0:
            for line in r.stdout.splitlines()[:15]:
                print(line)
        else:
            print("ERROR: nvidia-smi no disponible\n" + r.stderr[:200])
    except FileNotFoundError:
        print("ERROR: nvidia-smi no encontrado. ¿Drivers NVIDIA instalados?")

    # 2. PyTorch
    print(f"\n{SEP}\n2. PYTORCH + CUDA\n{SEP}")
    try:
        import torch
        print(f"  PyTorch version : {torch.__version__}")
        print(f"  CUDA disponible : {torch.cuda.is_available()}")
        print(f"  CUDA version    : {torch.version.cuda}")
        if torch.cuda.is_available():
            print(f"  GPU             : {torch.cuda.get_device_name(0)}")
            print(f"  VRAM            : {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        else:
            print("  CUDA no disponible para PyTorch")
    except ImportError:
        print("  ERROR: PyTorch no instalado")

    # 3. FFmpeg
    print(f"\n{SEP}\n3. FFMPEG — VERSIÓN Y ENCODERS\n{SEP}")
    try:
        r = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True, timeout=20)
        print(f"  {r.stdout.splitlines()[0] if r.stdout else '?'}")
        r2 = subprocess.run(["ffmpeg", "-encoders", "-v", "quiet"], capture_output=True, text=True, timeout=20)
        nvenc = [l.strip() for l in r2.stdout.splitlines() if "nvenc" in l.lower()]
        if nvenc:
            print("  Encoders NVENC encontrados:")
            for e in nvenc: print(f"    {e}")
        else:
            print("  ERROR: No se encontraron encoders NVENC")
    except FileNotFoundError:
        print("  ERROR: ffmpeg no encontrado en PATH")
        sys.exit(1)

    # 4. Test h264_nvenc real
    print(f"\n{SEP}\n4. TEST REAL h264_nvenc\n{SEP}")
    r = subprocess.run([
        "ffmpeg", "-y", "-v", "error", "-f", "lavfi",
        "-i", "color=black:size=128x128:rate=25",
        "-t", "0.2", "-an", "-pix_fmt", "yuv420p",
        "-c:v", "h264_nvenc", "-f", "null", "-"
    ], capture_output=True, text=True, timeout=30)
    if r.returncode == 0:
        print("  ✔  h264_nvenc funciona correctamente")
    else:
        print(f"  ✘  h264_nvenc falló (código {r.returncode})")
        print(f"  Error: {r.stderr.strip()[:400]}")

    # 5. nvcuda.dll
    print(f"\n{SEP}\n5. BÚSQUEDA DE nvcuda.dll EN EL SISTEMA\n{SEP}")
    found = False
    for p in [r"C:\Windows\System32\nvcuda.dll", r"C:\Windows\SysWOW64\nvcuda.dll"]:
        if os.path.exists(p):
            print(f"  ✔  Encontrado: {p}")
            found = True
    if not found:
        print("  ✘  nvcuda.dll NO encontrado en System32/SysWOW64")
        print("     Posible causa: drivers NVIDIA corruptos o no instalados")

    # 6. CUDA en PATH
    print(f"\n{SEP}\n6. RUTAS CUDA EN PATH DEL SISTEMA\n{SEP}")
    cuda_in_path = [p for p in os.environ.get("PATH", "").split(os.pathsep) if "cuda" in p.lower() or "nvidia" in p.lower()]
    if cuda_in_path:
        for p in cuda_in_path: print(f"  ✔  {p}")
    else:
        print("  ✘  No hay rutas CUDA/NVIDIA en el PATH")
        print("     Añade C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\vX.X\\bin al PATH")

    print(f"\n{SEP}\nDIAGNÓSTICO COMPLETADO\n{SEP}")
    print("Copia y pega la salida completa para continuar.\n")


# ── webcam ────────────────────────────────────────────────────────────────────

def cmd_webcam(args) -> None:
    try:
        import cv2
    except ImportError:
        print("ERROR: opencv-python no instalado.")
        sys.exit(1)

    print(f"Extrayendo frame en el segundo {args.second}...")
    frame_path = _extract_frame(args.file, args.second)
    frame      = cv2.imread(frame_path)
    os.unlink(frame_path)

    if frame is None:
        print("ERROR: No se pudo extraer el frame.")
        sys.exit(1)

    src_h, src_w = frame.shape[:2]
    print(f"Resolución: {src_w}x{src_h}\n")
    print("=" * 55)
    print("INSTRUCCIONES:")
    print("  1. Haz clic y arrastra sobre la WEBCAM")
    print("  2. Pulsa ENTER para confirmar")
    print("  3. Pulsa 'r' para repetir, 'q' para salir")
    print("=" * 55)

    scale     = min(1.0, 1280 / src_w)
    display_w = int(src_w * scale)
    display_h = int(src_h * scale)
    display   = cv2.resize(frame, (display_w, display_h))
    cv2.putText(display, "Selecciona zona WEBCAM + pulsa ENTER",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    roi = cv2.selectROI(
        "Selecciona zona WEBCAM - ENTER para confirmar",
        display, fromCenter=False, showCrosshair=True,
    )
    cv2.destroyAllWindows()

    if roi == (0, 0, 0, 0):
        print("Seleccion vacia. Ejecuta de nuevo.")
        sys.exit(1)

    rx, ry, rw, rh = int(roi[0] / scale), int(roi[1] / scale), int(roi[2] / scale), int(roi[3] / scale)
    w_ratio        = round(rw / src_w, 3)
    h_ratio        = round(rh / src_h, 3)
    y_center_ratio = round((ry + rh / 2) / src_h, 3)

    print("\n" + "=" * 55)
    print('COPIA ESTOS VALORES EN config.json -> "layout":')
    print("=" * 55)
    print(f'  "webcam_w_ratio":        {w_ratio},')
    print(f'  "webcam_h_ratio":        {h_ratio},')
    print(f'  "webcam_x_offset":       {rx},')
    print(f'  "webcam_y_center_ratio": {y_center_ratio},')
    print("=" * 55)
    print(f"Webcam: {rw}x{rh}px | {w_ratio*100:.1f}% ancho | {h_ratio*100:.1f}% alto")
    print(f"Centro vertical: {y_center_ratio*100:.1f}% del frame")

    result = frame.copy()
    cv2.rectangle(result, (rx, ry), (rx+rw, ry+rh), (0, 255, 0), 4)
    cv2.putText(result, "WEBCAM", (rx+5, ry+40), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
    cv2.imshow("Resultado - pulsa cualquier tecla para cerrar",
               cv2.resize(result, (display_w, display_h)))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ── border ────────────────────────────────────────────────────────────────────

def _measure_ratio(frame, cam_x, cam_y, cam_w, cam_h, hsv_lower, hsv_upper, border_px):
    import cv2
    import numpy as np

    h, w = frame.shape[:2]
    x1 = max(0, cam_x - border_px)
    y1 = max(0, cam_y - border_px)
    x2 = min(w, cam_x + cam_w + border_px)
    y2 = min(h, cam_y + cam_h + border_px)
    region = frame[y1:y2, x1:x2]

    inner_x1, inner_y1 = border_px, border_px
    inner_x2 = region.shape[1] - border_px
    inner_y2 = region.shape[0] - border_px

    mask = np.ones(region.shape[:2], dtype=np.uint8) * 255
    if inner_x2 > inner_x1 and inner_y2 > inner_y1:
        mask[inner_y1:inner_y2, inner_x1:inner_x2] = 0

    hsv           = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    purple        = cv2.inRange(hsv, np.array(hsv_lower, dtype=np.uint8), np.array(hsv_upper, dtype=np.uint8))
    purple_border = cv2.bitwise_and(purple, mask)

    border_pixels = int((mask > 0).sum())
    purple_pixels = int((purple_border > 0).sum())
    ratio         = purple_pixels / border_pixels if border_pixels > 0 else 0.0

    return ratio, purple_border, mask, region, (x1, y1, x2, y2)


def _analyze_frame(frame, config: dict, second: int, label: str = ""):
    import cv2
    import numpy as np

    layout = config.get("layout", {})
    cfg_f  = config.get("face_detection", {})
    src_h, src_w = frame.shape[:2]

    cam_w = int(src_w * layout.get("webcam_w_ratio", 0.137))
    cam_h = int(src_h * layout.get("webcam_h_ratio", 0.331))
    cam_x = layout.get("webcam_x_offset", 22)
    cam_y = max(0, min(int(src_h * layout.get("webcam_y_center_ratio", 0.5)) - cam_h // 2, src_h - cam_h))

    hsv_lower = cfg_f.get("border_color_hsv_lower", [125, 50, 50])
    hsv_upper = cfg_f.get("border_color_hsv_upper", [165, 255, 255])
    border_px = cfg_f.get("border_px", 8)
    min_ratio = cfg_f.get("border_min_ratio", 0.03)

    ratio, purple_border, _, _, (x1, y1, x2, y2) = _measure_ratio(
        frame, cam_x, cam_y, cam_w, cam_h, hsv_lower, hsv_upper, border_px
    )
    detected = ratio >= min_ratio

    print(f"\n{'='*55}")
    print(f"Segundo {second}s  {label}\n{'='*55}")
    print(f"  Zona webcam:       ({cam_x},{cam_y}) {cam_w}x{cam_h}px")
    print(f"  HSV lower/upper:   {hsv_lower} / {hsv_upper}")
    print(f"  border_px:         {border_px}\n")
    print(f"  >> border_min_ratio MEDIDO: {ratio:.4f}  ({ratio*100:.1f}%)\n")
    print(f"  border_min_ratio actual:    {min_ratio}")
    print(f"  Cámara detectada:           {'✓ SÍ' if detected else '✗ NO'}\n")
    if not detected and ratio > 0:
        print(f"  → Para detectar este frame: border_min_ratio <= {round(ratio * 0.8, 3)}")
    elif detected:
        print(f"  → Para rechazar este frame: border_min_ratio >= {round(ratio * 1.2, 3)}")
    print("=" * 55)

    vis = frame.copy()
    cv2.rectangle(vis, (cam_x, cam_y), (cam_x + cam_w, cam_y + cam_h), (0, 255, 0), 3)
    cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 100, 0), 2)
    purple_vis = np.zeros_like(frame)
    purple_vis[y1:y2, x1:x2][purple_border > 0] = [255, 0, 255]
    vis = cv2.addWeighted(vis, 0.8, purple_vis, 0.8, 0)
    color  = (0, 200, 0) if detected else (0, 0, 255)
    status = "DETECTADA" if detected else "NO detectada"
    cv2.putText(vis, f"s={second}  ratio={ratio:.3f}  [{status}]", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
    cv2.putText(vis, f"border_min_ratio actual: {min_ratio}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    return vis, ratio


def cmd_border(args) -> None:
    try:
        import cv2
    except ImportError:
        print("ERROR: opencv-python no instalado.")
        sys.exit(1)

    config  = json.loads(Path(args.config).read_text())
    seconds = [args.second] + ([args.second2] if args.second2 is not None else [])

    frames_vis = []
    for sec in seconds:
        print(f"Extrayendo frame en segundo {sec}s...")
        path  = _extract_frame(args.file, sec)
        frame = cv2.imread(path)
        os.unlink(path)
        if frame is None:
            print(f"ERROR: No se pudo extraer frame en segundo {sec}")
            continue
        label = "(con cámara?)" if sec == seconds[0] else "(sin cámara?)"
        vis, ratio = _analyze_frame(frame, config, sec, label)
        frames_vis.append((sec, vis, ratio))

    print("\nAbriendo ventana de visualización...")
    print("  Rosa/magenta = píxeles morados detectados")
    print("  Verde        = zona webcam configurada")
    print("  Azul         = zona de borde analizada")
    print("  Pulsa cualquier tecla para cerrar")

    for sec, vis, ratio in frames_vis:
        h, w  = vis.shape[:2]
        scale = min(1.0, 900 / w)
        small = cv2.resize(vis, (int(w * scale), int(h * scale)))
        cv2.imshow(f"Segundo {sec}s — ratio={ratio:.4f}", small)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Herramientas de calibración viral_clips",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    # gpu
    sub.add_parser("gpu", help="Diagnóstico completo de GPU")

    # webcam
    p_wc = sub.add_parser("webcam", help="Selección interactiva de zona webcam")
    p_wc.add_argument("--file",   required=True, help="Ruta al vídeo")
    p_wc.add_argument("--second", type=int, default=60, help="Segundo a extraer (default: 60)")

    # border
    p_bd = sub.add_parser("border", help="Medición del border_min_ratio")
    p_bd.add_argument("--file",    required=True, help="Ruta al vídeo")
    p_bd.add_argument("--second",  type=int, default=60, help="Primer segundo a analizar (default: 60)")
    p_bd.add_argument("--second2", type=int, default=None, help="Segundo adicional (opcional)")
    p_bd.add_argument("--config",  type=str, default="config.json", help="Ruta al config.json (default: config.json)")

    args = parser.parse_args()

    if args.cmd == "gpu":
        cmd_gpu(args)
    elif args.cmd == "webcam":
        cmd_webcam(args)
    elif args.cmd == "border":
        cmd_border(args)


if __name__ == "__main__":
    main()
