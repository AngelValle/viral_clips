"""
test_gpu.py
Diagnóstico completo de GPU para Windows.
Ejecutar desde la carpeta viral_clips con el entorno virtual activado:
    python test_gpu.py
"""

import subprocess
import sys
import os

SEP = "=" * 55

def run(cmd, **kwargs):
    return subprocess.run(cmd, capture_output=True, text=True,
                          timeout=20, **kwargs)

# ── 1. Drivers NVIDIA ─────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("1. DRIVERS NVIDIA (nvidia-smi)")
print(SEP)
try:
    r = run(["nvidia-smi"])
    if r.returncode == 0:
        for line in r.stdout.splitlines()[:15]:
            print(line)
    else:
        print("ERROR: nvidia-smi no disponible")
        print(r.stderr[:200])
except FileNotFoundError:
    print("ERROR: nvidia-smi no encontrado. ¿Drivers NVIDIA instalados?")

# ── 2. PyTorch ────────────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("2. PYTORCH + CUDA")
print(SEP)
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

# ── 3. FFmpeg — versión y encoders ────────────────────────────────────────────
print(f"\n{SEP}")
print("3. FFMPEG — VERSIÓN Y ENCODERS")
print(SEP)
try:
    r = run(["ffmpeg", "-version"])
    primera = r.stdout.splitlines()[0] if r.stdout else "?"
    print(f"  {primera}")

    r2 = run(["ffmpeg", "-encoders", "-v", "quiet"])
    nvenc = [l.strip() for l in r2.stdout.splitlines() if "nvenc" in l.lower()]
    if nvenc:
        print(f"  Encoders NVENC encontrados:")
        for e in nvenc:
            print(f"    {e}")
    else:
        print("  ERROR: No se encontraron encoders NVENC")
except FileNotFoundError:
    print("  ERROR: ffmpeg no encontrado en PATH")
    sys.exit(1)

# ── 4. Test h264_nvenc real ───────────────────────────────────────────────────
print(f"\n{SEP}")
print("4. TEST REAL h264_nvenc")
print(SEP)
r = subprocess.run(
    [
        "ffmpeg", "-y", "-v", "error",
        "-f", "lavfi",
        "-i", "color=black:size=128x128:rate=25",
        "-t", "0.2",
        "-an",
        "-pix_fmt", "yuv420p",
        "-c:v", "h264_nvenc",
        "-f", "null", "-"
    ],
    capture_output=True, text=True, timeout=30
)
if r.returncode == 0:
    print("  ✔  h264_nvenc funciona correctamente")
else:
    print(f"  ✘  h264_nvenc falló (código {r.returncode})")
    print(f"  Error: {r.stderr.strip()[:400]}")

# ── 5. Test con nvcuda.dll / libcuda ─────────────────────────────────────────
print(f"\n{SEP}")
print("5. BÚSQUEDA DE nvcuda.dll EN EL SISTEMA")
print(SEP)
cuda_paths = [
    r"C:\Windows\System32\nvcuda.dll",
    r"C:\Windows\SysWOW64\nvcuda.dll",
]
found = False
for p in cuda_paths:
    if os.path.exists(p):
        print(f"  ✔  Encontrado: {p}")
        found = True
if not found:
    print("  ✘  nvcuda.dll NO encontrado en System32/SysWOW64")
    print("     Posible causa: drivers NVIDIA corruptos o no instalados")

# ── 6. Variable de entorno PATH relevante ────────────────────────────────────
print(f"\n{SEP}")
print("6. RUTAS CUDA EN PATH DEL SISTEMA")
print(SEP)
path_entries = os.environ.get("PATH", "").split(os.pathsep)
cuda_in_path = [p for p in path_entries if "cuda" in p.lower() or "nvidia" in p.lower()]
if cuda_in_path:
    for p in cuda_in_path:
        print(f"  ✔  {p}")
else:
    print("  ✘  No hay rutas CUDA/NVIDIA en el PATH")
    print("     Añade C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\vX.X\\bin al PATH")

print(f"\n{SEP}")
print("DIAGNÓSTICO COMPLETADO")
print(SEP)
print("Copia y pega la salida completa para continuar.\n")