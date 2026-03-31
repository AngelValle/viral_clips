"""
scanner.py
Escanea carpetas locales en busca de vídeos válidos y construye
una cola de procesamiento ordenada por fecha (más reciente primero).
"""

import os
import json
import logging
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)


def load_config(config_path: str = "config.json") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def is_valid_video(path: Path, min_size_mb: float = 1.0) -> bool:
    """Comprueba que el fichero exista, tenga extensión soportada y tamaño mínimo."""
    if not path.is_file():
        return False
    if path.suffix.lower() not in (".mp4", ".mkv", ".mov"):
        return False
    if path.stat().st_size < min_size_mb * 1024 * 1024:
        logger.warning(f"Fichero demasiado pequeño, se omite: {path.name}")
        return False
    return True


def scan_folder(folder: Path, supported_formats: List[str]) -> List[Path]:
    """
    Devuelve una lista de vídeos válidos dentro de folder (recursivo),
    ordenados de más reciente a más antiguo.
    """
    candidates = []
    for ext in supported_formats:
        candidates.extend(folder.rglob(f"*{ext}"))

    valid = [p for p in candidates if is_valid_video(p)]
    valid.sort(key=lambda p: p.stat().st_mtime, reverse=True)

    logger.info(f"Scanner: {len(valid)} vídeo(s) encontrado(s) en '{folder}'")
    return valid


def already_processed(video_path: Path, processed_dir: Path) -> bool:
    """Comprueba si el vídeo ya fue procesado (existe en processed_dir)."""
    return (processed_dir / video_path.name).exists()


def build_queue(config: dict, single_file: Path = None) -> List[Path]:
    """
    Construye la cola de procesamiento según configuración.
    Si single_file se especifica, la cola contiene solo ese fichero.
    """
    processed_dir = Path(config["paths"]["processed_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)

    if single_file:
        if not is_valid_video(single_file):
            raise ValueError(f"El fichero no es un vídeo válido: {single_file}")
        return [single_file]

    input_dir = Path(config["paths"]["input_dir"])
    if not input_dir.exists():
        raise FileNotFoundError(f"La carpeta de entrada no existe: {input_dir}")

    all_videos = scan_folder(input_dir, config["paths"]["supported_formats"])
    queue = [v for v in all_videos if not already_processed(v, processed_dir)]

    if not queue:
        logger.info("No hay vídeos nuevos pendientes de procesar.")
    return queue


def mark_as_processed(video_path: Path, processed_dir: Path) -> None:
    """Mueve el vídeo original a la carpeta de procesados."""
    import shutil
    dest = processed_dir / video_path.name
    shutil.move(str(video_path), str(dest))
    logger.info(f"Movido a procesados: {video_path.name}")
