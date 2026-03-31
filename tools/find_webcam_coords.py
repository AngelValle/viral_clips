"""
tools/find_webcam_coords.py
Extrae un frame del vídeo y abre una ventana interactiva donde puedes
hacer clic y arrastrar para seleccionar la zona de la webcam.
Al terminar imprime los valores exactos para pegar en config.json.

Uso:
    python tools/find_webcam_coords.py --file videos/input/tu_video.mp4
    python tools/find_webcam_coords.py --file videos/input/tu_video.mp4 --second 30
"""

import argparse
import sys
import subprocess
import tempfile
import os


def extract_frame(video_path: str, second: int = 60) -> str:
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    tmp.close()
    subprocess.run([
        "ffmpeg", "-y", "-ss", str(second),
        "-i", video_path,
        "-vframes", "1",
        "-q:v", "2",
        tmp.name,
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return tmp.name


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file",   required=True, help="Ruta al vídeo")
    parser.add_argument("--second", type=int, default=60,
                        help="Segundo del vídeo a extraer (default: 60)")
    args = parser.parse_args()

    try:
        import cv2
    except ImportError:
        print("ERROR: opencv-python no instalado.")
        sys.exit(1)

    print(f"Extrayendo frame en el segundo {args.second}...")
    frame_path = extract_frame(args.file, args.second)

    frame = cv2.imread(frame_path)
    os.unlink(frame_path)

    if frame is None:
        print("ERROR: No se pudo extraer el frame.")
        sys.exit(1)

    src_h, src_w = frame.shape[:2]
    print(f"Resolución: {src_w}x{src_h}")
    print()
    print("=" * 55)
    print("INSTRUCCIONES:")
    print("  1. Haz clic y arrastra sobre la WEBCAM")
    print("  2. Pulsa ENTER para confirmar")
    print("  3. Pulsa 'r' para repetir, 'q' para salir")
    print("=" * 55)

    max_display = 1280
    scale       = min(1.0, max_display / src_w)
    display_w   = int(src_w * scale)
    display_h   = int(src_h * scale)
    display     = cv2.resize(frame, (display_w, display_h))

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

    rx = int(roi[0] / scale)
    ry = int(roi[1] / scale)
    rw = int(roi[2] / scale)
    rh = int(roi[3] / scale)

    w_ratio         = round(rw / src_w, 3)
    h_ratio         = round(rh / src_h, 3)
    x_offset        = rx
    y_center_ratio  = round((ry + rh / 2) / src_h, 3)

    print()
    print("=" * 55)
    print("COPIA ESTOS VALORES EN config.json -> \"layout\":")
    print("=" * 55)
    print(f'  "webcam_w_ratio":        {w_ratio},')
    print(f'  "webcam_h_ratio":        {h_ratio},')
    print(f'  "webcam_x_offset":       {x_offset},')
    print(f'  "webcam_y_center_ratio": {y_center_ratio},')
    print("=" * 55)
    print(f"Webcam: {rw}x{rh}px | {w_ratio*100:.1f}% ancho | {h_ratio*100:.1f}% alto")
    print(f"Centro vertical: {y_center_ratio*100:.1f}% del frame")

    # Mostrar resultado
    result = frame.copy()
    cv2.rectangle(result, (rx, ry), (rx+rw, ry+rh), (0, 255, 0), 4)
    cv2.putText(result, "WEBCAM", (rx+5, ry+40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
    cv2.imshow("Resultado - pulsa cualquier tecla para cerrar",
               cv2.resize(result, (display_w, display_h)))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
