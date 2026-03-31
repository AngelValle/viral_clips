"""
gpu_utils.py
Detección de GPU para Windows 11 con NVIDIA RTX.
Jerarquía: CUDA → Intel QSV → AMD AMF → CPU fallback.
MPS (Apple) y ROCm (AMD Linux) eliminados — no aplican en este sistema.
"""

import logging
import subprocess
from functools import lru_cache

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_torch_device() -> str:
    """
    Detecta el mejor dispositivo PyTorch disponible.
    En Windows 11 con RTX 4080 SUPER siempre devuelve 'cuda'.
    """
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            logger.info(f"GPU [CUDA]: {name} ({vram:.1f} GB VRAM)")
            return "cuda"
    except ImportError:
        logger.warning("PyTorch no instalado. Usando CPU.")
    logger.warning("CUDA no disponible. Usando CPU.")
    return "cpu"


@lru_cache(maxsize=1)
def get_ffmpeg_hwaccel() -> dict:
    """
    Detecta el encoder hardware disponible en FFmpeg.
    Prioridad para Windows 11 NVIDIA: h264_nvenc → h264_amf → h264_qsv → libx264.
    """
    try:
        result = subprocess.run(
            ["ffmpeg", "-encoders", "-v", "quiet"],
            capture_output=True, text=True, timeout=10
        )
        encoders_output = result.stdout
    except (FileNotFoundError, subprocess.TimeoutExpired):
        logger.error("ffmpeg no encontrado.")
        return _sw_fallback()

    candidates = [
        {
            "encoder":        "h264_nvenc",
            "hwaccel":        "cuda",
            "extra_enc_args": ["-rc", "vbr", "-cq", "23", "-preset", "p4"],
            "is_hw":          True,
            "label":          "NVIDIA NVENC",
        },
        {
            "encoder":        "h264_amf",
            "hwaccel":        "d3d11va",
            "extra_enc_args": ["-quality", "balanced"],
            "is_hw":          True,
            "label":          "AMD AMF",
        },
        {
            "encoder":        "h264_qsv",
            "hwaccel":        "qsv",
            "extra_enc_args": ["-preset", "medium"],
            "is_hw":          True,
            "label":          "Intel Quick Sync",
        },
    ]

    for c in candidates:
        if c["encoder"] in encoders_output and _test_ffmpeg_encoder(c["encoder"]):
            logger.info(f"Encoder FFmpeg: {c['label']} ({c['encoder']})")
            return c

    logger.warning("Sin encoder hardware. Usando libx264 (CPU).")
    return _sw_fallback()


def _sw_fallback() -> dict:
    return {
        "encoder":        "libx264",
        "hwaccel":        None,
        "extra_enc_args": ["-preset", "fast", "-crf", "23"],
        "is_hw":          False,
        "label":          "libx264 (CPU)",
    }


def _test_ffmpeg_encoder(encoder: str) -> bool:
    try:
        result = subprocess.run(
            ["ffmpeg", "-y", "-v", "error",
             "-f", "lavfi", "-i", "color=black:size=256x256:rate=25",
             "-t", "0.2", "-an", "-pix_fmt", "yuv420p",
             "-c:v", encoder, "-f", "null", "-"],
            capture_output=True, timeout=20
        )
        return result.returncode == 0
    except Exception:
        return False


@lru_cache(maxsize=1)
def get_cv2_device() -> str:
    """Comprueba si OpenCV fue compilado con soporte CUDA."""
    try:
        import cv2
        info = cv2.getBuildInformation()
        if "CUDA" in info and "YES" in info.split("CUDA")[1][:100]:
            logger.info("OpenCV con soporte CUDA.")
            return "cuda"
    except Exception:
        pass
    return "cpu"


def print_gpu_summary() -> None:
    hw = get_ffmpeg_hwaccel()
    logger.info(f"PyTorch: {get_torch_device()} | FFmpeg: {hw['encoder']} | OpenCV: {get_cv2_device()}")


def free_gpu_memory() -> None:
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
