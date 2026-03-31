#!/usr/bin/env python3
"""
watch.py
Modo vigilancia: monitoriza la carpeta input/ y procesa automáticamente
cualquier vídeo nuevo que aparezca (.mp4, .mkv, .mov).

Uso:
    python watch.py
    python watch.py --config mi_config.json
"""

import argparse
import logging
import sys
import time
from pathlib import Path

from watchdog.observers import Observer
from watchdog.events    import FileSystemEventHandler, FileCreatedEvent

from modules.scanner    import load_config, is_valid_video, mark_as_processed
from main               import process_video

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("watch")


class VideoHandler(FileSystemEventHandler):
    """Reacciona a la creación de nuevos ficheros de vídeo en la carpeta vigilada."""

    def __init__(self, config: dict):
        super().__init__()
        self.config       = config
        self.processed_dir = Path(config["paths"]["processed_dir"])
        self.supported    = set(config["paths"]["supported_formats"])
        self._processing  = set()  # evita doble procesado durante escritura

    def on_created(self, event: FileCreatedEvent):
        if event.is_directory:
            return
        path = Path(event.src_path)
        if path.suffix.lower() not in self.supported:
            return
        if str(path) in self._processing:
            return

        # Espera a que el fichero termine de escribirse (tamaño estable)
        self._wait_until_stable(path)
        if not is_valid_video(path):
            return

        logger.info(f"[WATCH] Nuevo vídeo detectado: {path.name}")
        self._processing.add(str(path))
        try:
            clips = process_video(path, self.config)
            if clips and self.config["watch_mode"].get("move_processed", True):
                mark_as_processed(path, self.processed_dir)
            logger.info(f"[WATCH] Clips generados para {path.name}: {len(clips)}")
        except Exception as e:
            logger.error(f"[WATCH] Error procesando {path.name}: {e}")
        finally:
            self._processing.discard(str(path))

    @staticmethod
    def _wait_until_stable(path: Path, checks: int = 5, interval: float = 2.0):
        """Espera hasta que el tamaño del fichero no cambie entre comprobaciones."""
        prev_size = -1
        for _ in range(checks):
            try:
                size = path.stat().st_size
            except FileNotFoundError:
                return
            if size == prev_size and size > 0:
                return
            prev_size = size
            time.sleep(interval)


def parse_args():
    parser = argparse.ArgumentParser(description="Viral Clip Watcher")
    parser.add_argument("--config", type=str, default="config.json")
    return parser.parse_args()


def main():
    args   = parse_args()
    config = load_config(args.config)

    input_dir = Path(config["paths"]["input_dir"])
    input_dir.mkdir(parents=True, exist_ok=True)

    poll_interval = config["watch_mode"].get("poll_interval_seconds", 10)
    logger.info(f"Vigilando carpeta: {input_dir.resolve()}")
    logger.info("Pulsa Ctrl+C para detener.")

    handler  = VideoHandler(config)
    observer = Observer()
    observer.schedule(handler, str(input_dir), recursive=True)
    observer.start()

    try:
        while True:
            time.sleep(poll_interval)
    except KeyboardInterrupt:
        logger.info("Deteniendo vigilancia...")
        observer.stop()
    observer.join()


if __name__ == "__main__":
    main()
