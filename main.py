#!/usr/bin/env python3
"""
main.py
Punto de entrada del pipeline de extracción de clips virales.
Incluye modo watcher automático (--watch) y logging visual con colores ANSI.
"""

import argparse
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import threading
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", message="Failed to launch Triton kernels")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")


# ── Logging visual ────────────────────────────────────────────────────────────

class C:
    RESET   = "\033[0m"
    BOLD    = "\033[1m"
    DIM     = "\033[2m"
    WHITE   = "\033[97m"
    GREY    = "\033[90m"
    CYAN    = "\033[96m"
    GREEN   = "\033[92m"
    YELLOW  = "\033[93m"
    RED     = "\033[91m"
    MAGENTA = "\033[95m"
    BLUE    = "\033[94m"
    ORANGE  = "\033[38;5;208m"
    BG_DARK = "\033[48;5;235m"


_MODULE_ABBR = {
    "main":                        "main",
    "modules.transcriber":         "whisper",
    "modules.viral_scorer":        "scorer",
    "modules.face_detector":       "faces",
    "modules.composer":            "composer",
    "modules.exporter":            "exporter",
    "modules.censor":              "censor",
    "modules.cache":               "cache",
    "modules.scanner":             "scanner",
    "modules.gpu_utils":           "gpu",
}


def _abbr(name: str) -> str:
    return _MODULE_ABBR.get(name, name.split(".")[-1])


class PipelineFormatter(logging.Formatter):
    LEVEL_STYLE = {
        logging.DEBUG:    (C.GREY,          "◌"),
        logging.INFO:     (C.CYAN,          "✦"),
        logging.WARNING:  (C.YELLOW,        "⚠"),
        logging.ERROR:    (C.RED,           "✖"),
        logging.CRITICAL: (C.RED + C.BOLD,  "✖"),
    }

    _RE_PASO   = re.compile(r"^PASO\s+(\d+)/(\d+)\s+[—–-]\s+(.+)$")
    _RE_CLIP   = re.compile(r"^Clip\s+(\d+)/(\d+)\s+\[(.+?)\]\s+score=([\d.]+)$")
    _RE_OK     = re.compile(r"^[✔✓☑]\s+(.+)$")
    _RE_CACHED = re.compile(r"^(PASO\s+\d+/\d+\s+[—–-]\s+.+?)\s+✓\s+\(desde caché\)$")
    _RE_SEP    = re.compile(r"^[═─=\-]{10,}$")

    def format(self, record: logging.LogRecord) -> str:
        color, icon = self.LEVEL_STYLE.get(record.levelno, (C.WHITE, "·"))
        ts     = self.formatTime(record, "%H:%M:%S")
        module = _abbr(record.name)
        msg    = record.getMessage()

        if self._RE_SEP.match(msg):
            return f"{C.DIM}{C.GREY}{'─' * 60}{C.RESET}"

        if msg.startswith("Procesando:"):
            name = msg.replace("Procesando:", "").strip()
            bar  = "═" * 58
            return (
                f"\n{C.BOLD}{C.MAGENTA}{bar}{C.RESET}\n"
                f"  {C.BOLD}{C.WHITE}▶  {name}{C.RESET}\n"
                f"{C.BOLD}{C.MAGENTA}{bar}{C.RESET}"
            )

        m = self._RE_PASO.match(msg)
        if m:
            n, total, title = m.group(1), m.group(2), m.group(3)
            badge = f"{C.BOLD}{C.BG_DARK}{C.CYAN} {n}/{total} {C.RESET}"
            return f"\n{C.DIM}{C.GREY}{ts}{C.RESET}  {badge}  {C.BOLD}{C.WHITE}{title}{C.RESET}"

        m = self._RE_CACHED.match(msg)
        if m:
            inner = self._RE_PASO.match(m.group(1))
            if inner:
                n, total, title = inner.group(1), inner.group(2), inner.group(3)
                badge  = f"{C.BOLD}{C.BG_DARK}{C.GREY} {n}/{total} {C.RESET}"
                cached = f"{C.DIM}{C.GREEN}  ↩ caché{C.RESET}"
                return f"\n{C.DIM}{C.GREY}{ts}{C.RESET}  {badge}  {C.DIM}{C.WHITE}{title}{cached}{C.RESET}"

        m = self._RE_CLIP.match(msg)
        if m:
            n, total, rng, score = m.group(1), m.group(2), m.group(3), m.group(4)
            return (
                f"\n{C.DIM}{C.GREY}{'·' * 60}{C.RESET}\n"
                f"  {C.BOLD}{C.ORANGE}Clip {n}{C.RESET}{C.DIM}{C.GREY}/{total}{C.RESET}"
                f"  {C.YELLOW}[{rng}]{C.RESET}"
                f"  {C.DIM}score {C.CYAN}{score}{C.RESET}"
            )

        m = self._RE_OK.match(msg)
        if m:
            return (
                f"{C.DIM}{C.GREY}{ts}{C.RESET}  "
                f"{C.BOLD}{C.GREEN}✔{C.RESET}  {C.GREEN}{m.group(1)}{C.RESET}"
            )

        if record.levelno == logging.WARNING:
            return (
                f"{C.DIM}{C.GREY}{ts}{C.RESET}  "
                f"{C.YELLOW}⚠{C.RESET}  {C.YELLOW}{msg}{C.RESET}"
                f"  {C.DIM}{C.GREY}[{module}]{C.RESET}"
            )

        if record.levelno >= logging.ERROR:
            return (
                f"{C.DIM}{C.GREY}{ts}{C.RESET}  "
                f"{C.BOLD}{C.RED}✖{C.RESET}  {C.BOLD}{C.RED}{msg}{C.RESET}"
                f"  {C.DIM}{C.GREY}[{module}]{C.RESET}"
            )

        if msg.startswith("  ") or re.match(r"^\[\d+/\d+\]", msg):
            return (
                f"{C.DIM}{C.GREY}{ts}{C.RESET}  "
                f"{C.DIM}{C.GREY}│{C.RESET}  {C.DIM}{msg}{C.RESET}"
            )

        return (
            f"{C.DIM}{C.GREY}{ts}{C.RESET}  "
            f"{color}{icon}{C.RESET}  {msg}"
            f"  {C.DIM}{C.GREY}[{module}]{C.RESET}"
        )


def setup_logging(level: int = logging.INFO) -> None:
    if sys.platform == "win32":
        try:
            import ctypes
            ctypes.windll.kernel32.SetConsoleMode(ctypes.windll.kernel32.GetStdHandle(-11), 7)
        except Exception:
            pass

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(PipelineFormatter())

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)

    for noisy in ("faster_whisper", "torch", "PIL", "urllib3", "httpx", "httpcore", "anthropic"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


# ── Pipeline modules ──────────────────────────────────────────────────────────

from modules.scanner      import build_queue, load_config, mark_as_processed, is_valid_video
from modules.transcriber  import transcribe, words_in_range, generate_subtitles
from modules.viral_scorer import detect_viral_moments
from modules.exporter     import (remap_words_to_fragments, extract_segment,
                                   embed_subtitles, save_metadata, generate_clip_metadata)
from modules.face_detector import analyze_video_faces
from modules.censor        import build_profanity_set, detect_profanity, apply_audio_censorship
from modules.composer      import compose_dynamic
from modules.gpu_utils     import print_gpu_summary
from modules.cache         import PipelineCache

setup_logging()
logger = logging.getLogger("main")


# ── Helpers ───────────────────────────────────────────────────────────────────

def get_video_duration(video_path: Path) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", str(video_path)],
        capture_output=True, text=True,
    )
    return float(json.loads(result.stdout)["format"]["duration"])


def clip_name_from_pattern(pattern: str, source_name: str, n: int) -> str:
    return pattern.format(source_name=source_name, n=str(n).zfill(2))


def _shift_words(words: list, delta: float) -> list:
    if abs(delta) < 0.01:
        return words
    return [
        {**w,
         "start": max(0.0, round(w["start"] + delta, 3)),
         "end":   max(0.0, round(w["end"]   + delta, 3))}
        for w in words
    ]


# ── Pipeline principal ────────────────────────────────────────────────────────

def process_video(video_path: Path, config: dict, use_cache: bool = True, max_step: int = 8) -> list:
    logger.info(f"Procesando: {video_path.name}")

    clips_dir = Path(config["paths"]["output_dir"]) / "clips"
    meta_dir  = Path(config["paths"]["output_dir"]) / "metadata"
    clips_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    cache = PipelineCache(video_path, config) if use_cache else None

    # ── PASO 1: Transcripción ─────────────────────────────────────────────────
    if max_step < 1:
        return []
    all_words = cache.get_transcription() if cache else None
    if all_words is None:
        logger.info("PASO 1/8 — Transcripción con Whisper")
        all_words = transcribe(video_path, config)
        if cache:
            cache.save_transcription(all_words)
    else:
        logger.info("PASO 1/8 — Transcripción ✓ (desde caché)")

    duration = get_video_duration(video_path)

    # ── PASO 2: Detección de momentos virales ─────────────────────────────────
    if max_step < 2:
        return []
    segments = cache.get_segments() if cache else None
    if segments is None:
        logger.info("PASO 2/8 — Detección de momentos virales")
        segments = detect_viral_moments(video_path, all_words, config, duration)
        if cache:
            cache.save_segments(segments)
    else:
        logger.info("PASO 2/8 — Segmentos virales ✓ (desde caché)")

    if not segments:
        logger.warning(f"No se detectaron segmentos virales en: {video_path.name}")
        return []

    if max_step < 3:
        logger.info(f"⏸ Ejecución pausada en el Paso 2 para {video_path.name} (según el límite max-step).")
        return []

    # ── PASOS 3–8: Procesamiento por segmento ─────────────────────────────────
    generated_clips = []
    profanity_set   = build_profanity_set(config)
    source_name     = video_path.stem
    naming_pattern  = config["output"]["naming_pattern"]

    for idx, segment in enumerate(segments, start=1):
        logger.info(
            f"Clip {idx}/{len(segments)} "
            f"[{segment['start']:.1f}s – {segment['end']:.1f}s] "
            f"score={segment.get('score', 0):.3f}"
        )

        clip_stem  = Path(clip_name_from_pattern(naming_pattern, source_name, idx)).stem
        final_path = clips_dir / f"{clip_stem}.mp4"

        if cache and cache.is_clip_done(final_path) and max_step >= 8:
            logger.info(f"Clip {idx} ✓ ya existe, saltando ({final_path.name})")
            generated_clips.append(final_path)
            continue

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)

            # ── PASO 3: Extracción del segmento crudo ─────────────────────────
            if max_step < 3:
                continue
            logger.info("PASO 3/8 — Extrayendo segmento crudo")
            raw_clip = tmp / "raw.mp4"
            extract_segment(video_path, segment["start"], segment["end"],
                            raw_clip, fragments=segment.get("fragments"))

            # ── PASO 4: Detección facial ──────────────────────────────────────
            if max_step < 4:
                continue
            face_enabled   = config.get("face_detection", {}).get("enabled", True)
            detect_driving = config.get("layout", {}).get("detect_driving", True)
            if not face_enabled:
                logger.info("PASO 4/8 — Detección facial deshabilitada (fullscreen mode)")
                face_data   = []
                is_driving  = False
            else:
                logger.info("PASO 4/8 — Detección facial")
                face_data = cache.get_face_data(clip_stem) if cache else None
                if face_data is None:
                    face_data, is_driving = analyze_video_faces(raw_clip, config)
                    if not detect_driving:
                        is_driving = False
                    if cache:
                        cache.save_face_data(clip_stem, face_data, is_driving=is_driving)
                else:
                    logger.info("PASO 4/8 — Face data ✓ (desde caché)")
                    is_driving = (cache.get_is_driving(clip_stem) if cache else False)
                    if not detect_driving:
                        is_driving = False

            # ── PASO 5: Composición 9:16 (omitir si formato horizontal) ──────
            if max_step < 5:
                continue
            orientation = config.get("output", {}).get("orientation", "vertical")
            composed    = tmp / "composed.mp4"
            if orientation == "horizontal":
                logger.info("PASO 5/8 — Formato horizontal, omitiendo composición 9:16")
                shutil.copy2(raw_clip, composed)
            else:
                logger.info("PASO 5/8 — Composición 9:16 dinámica")
                compose_dynamic(raw_clip, face_data, config, composed,
                                is_driving=is_driving)

            # ── PASO 6: Censura + subtítulos + render final ───────────────────
            if max_step < 6:
                continue
            logger.info("PASO 6/8 — Censura, subtítulos y render final")
            clip_words = words_in_range(all_words, segment["start"], segment["end"])

            fragments = segment.get("fragments")
            if fragments and len(fragments) > 1:
                clip_words_rel = remap_words_to_fragments(clip_words, fragments)
            else:
                offset = segment["start"]
                clip_words_rel = [
                    {**w, "start": round(w["start"] - offset, 3),
                           "end":   round(w["end"]   - offset, 3)}
                    for w in clip_words
                ]

            censored_words = detect_profanity(clip_words_rel, profanity_set)
            censored_audio = tmp / "censored.mp4"
            apply_audio_censorship(composed, censored_audio, censored_words, config)

            sub_offset = config.get("subtitles", {}).get("manual_offset_sec", 0.0)
            sub_words  = _shift_words(censored_words, sub_offset) if sub_offset else censored_words

            try:
                _probe    = subprocess.run(
                    ["ffprobe", "-v", "quiet", "-print_format", "json",
                     "-show_format", str(censored_audio)],
                    capture_output=True, text=True,
                )
                _clip_dur = float(json.loads(_probe.stdout)["format"]["duration"])
            except Exception:
                _clip_dur = segment["end"] - segment["start"]

            sub_words = [
                {**w,
                 "start": min(w["start"], _clip_dur),
                 "end":   min(w["end"],   _clip_dur)}
                for w in sub_words
                if w["start"] < _clip_dur
            ]

            # Para horizontal, sobrescribir la resolución con la real del vídeo compuesto
            if orientation == "horizontal":
                try:
                    _vp = subprocess.run(
                        ["ffprobe", "-v", "quiet", "-print_format", "json",
                         "-show_streams", "-select_streams", "v:0", str(composed)],
                        capture_output=True, text=True,
                    )
                    _vs = json.loads(_vp.stdout)["streams"][0]
                    _eff_out = {**config["output"],
                                "resolution_w": _vs["width"],
                                "resolution_h": _vs["height"]}
                    _eff_config = {**config, "output": _eff_out}
                except Exception:
                    _eff_config = config
            else:
                _eff_config = config

            if config.get("subtitles", {}).get("enabled", True):
                ass_file = generate_subtitles(sub_words, _eff_config, tmp, clip_stem)
                embed_subtitles(censored_audio, ass_file, final_path, _eff_config)
            else:
                logger.info("PASO 6/8 — Subtítulos deshabilitados, copiando sin .ass")
                shutil.copy2(censored_audio, final_path)
            save_metadata(final_path, video_path, segment,
                          censored_words, config, meta_dir)

            # ── PASO 7: Metadata viral IA ─────────────────────────────────────
            if max_step < 7:
                continue
            if os.environ.get("GEMINI_API_KEY", "").strip():
                logger.info("PASO 7/8 — Generando títulos y descripciones con IA")
                transcript_text = " ".join(
                    w.get("censored_text", w["word"]) for w in censored_words
                )
                generate_clip_metadata(
                    transcript  = transcript_text,
                    viral_score = segment.get("score", 0),
                    duration    = segment["end"] - segment["start"],
                    clip_name   = clip_stem,
                    config      = config,
                    output_dir  = meta_dir,
                )
            else:
                logger.info("PASO 7/8 — Metadata IA omitida (sin GEMINI_API_KEY)")

            # ── PASO 8: Auto-publicación ──────────────────────────────────────
            if max_step < 8:
                continue
            logger.info("PASO 8/8 — Auto-Publicación (YouTube/TikTok)")
            meta_json = meta_dir / f"{clip_stem}.json"
            if meta_json.exists():
                with open(meta_json, "r", encoding="utf-8") as f:
                    meta_data = json.load(f)
                if meta_data.get("publish_yt"):
                    try:
                        from modules.publisher import publish_to_youtube
                        publish_to_youtube(final_path, meta_json)
                    except ImportError:
                        logger.error("Publisher module no encontrado.")
                if meta_data.get("publish_tk"):
                    try:
                        from modules.publisher import publish_to_tiktok
                        publish_to_tiktok(final_path, meta_json)
                    except ImportError:
                        logger.error("Publisher module no encontrado.")

        logger.info(f"✔ Clip generado: {final_path.name}")
        generated_clips.append(final_path)

    return generated_clips


# ── Watch mode ────────────────────────────────────────────────────────────────

class VideoHandler:
    """Reacciona a la creación de nuevos ficheros de vídeo en la carpeta vigilada."""

    def __init__(self, config: dict, max_step: int = 8):
        self.config        = config
        self.max_step      = max_step
        self.processed_dir = Path(config["paths"]["processed_dir"])
        self.supported     = set(config["paths"]["supported_formats"])
        self._processing   = set()

    def on_created(self, event) -> None:
        if event.is_directory:
            return
        path = Path(event.src_path)
        if path.suffix.lower() not in self.supported:
            return
        if str(path) in self._processing:
            return

        self._wait_until_stable(path)
        if not is_valid_video(path):
            return

        logger.info(f"[WATCH] Nuevo vídeo detectado: {path.name}")
        self._processing.add(str(path))
        try:
            clips = process_video(path, self.config, max_step=self.max_step)
            if clips and self.config["watch_mode"].get("move_processed", True) and self.max_step == 8:
                mark_as_processed(path, self.processed_dir)
            logger.info(f"[WATCH] Clips generados para {path.name}: {len(clips)}")
        except Exception as e:
            logger.error(f"[WATCH] Error procesando {path.name}: {e}")
        finally:
            self._processing.discard(str(path))

    @staticmethod
    def _wait_until_stable(path: Path, checks: int = 5, interval: float = 2.0) -> None:
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


def _run_watch(config: dict, max_step: int) -> None:
    from watchdog.observers import Observer
    from watchdog.events    import FileSystemEventHandler

    input_dir = Path(config["paths"]["input_dir"])
    input_dir.mkdir(parents=True, exist_ok=True)

    poll_interval = config["watch_mode"].get("poll_interval_seconds", 10)
    logger.info(f"Vigilando carpeta: {input_dir.resolve()}")
    logger.info("Pulsa Ctrl+C para detener.")

    handler_obj = VideoHandler(config, max_step=max_step)

    class _Adapter(FileSystemEventHandler):
        def on_created(self, event):
            handler_obj.on_created(event)

    observer = Observer()
    observer.schedule(_Adapter(), str(input_dir), recursive=True)
    observer.start()

    try:
        while True:
            time.sleep(poll_interval)
    except KeyboardInterrupt:
        logger.info("Deteniendo vigilancia...")
        observer.stop()
    observer.join()


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Viral Clip Automation — procesa vídeos locales."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--file",   type=str,        help="Ruta a un único fichero de vídeo.")
    group.add_argument("--folder", type=str,        help="Ruta a carpeta de vídeos.")
    group.add_argument("--watch",  action="store_true", help="Modo vigilancia: monitoriza input/ automáticamente.")

    script_dir = Path(__file__).parent
    default_config = str(script_dir / "config.json")
    parser.add_argument("--config",      type=str, default=default_config)
    parser.add_argument("--no-move",     action="store_true",
                        help="No mover los vídeos a 'processed' al terminar.")
    parser.add_argument("--no-cache",    action="store_true",
                        help="Ignorar caché y reprocesar todo desde cero.")
    parser.add_argument("--clear-cache", action="store_true",
                        help="Borrar la caché del vídeo antes de procesar.")
    parser.add_argument("--max-step",    type=int, default=8,
                        help="Paso máximo a ejecutar (1-8, default=8).")
    return parser.parse_args()


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    config = load_config(args.config)
    print_gpu_summary()

    processed_dir = Path(config["paths"]["processed_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)

    if args.watch:
        _run_watch(config, max_step=args.max_step)
        return

    move_after = not args.no_move and config["watch_mode"].get("move_processed", True)
    use_cache  = not args.no_cache

    if args.file:
        video_path = Path(args.file)
        if args.clear_cache:
            PipelineCache(video_path, config).clear()
        clips = process_video(video_path, config, use_cache=use_cache, max_step=args.max_step)
        if clips and move_after and args.max_step == 8:
            mark_as_processed(video_path, processed_dir)
    else:
        queue = build_queue(config)
        if not queue:
            logger.info("No hay vídeos nuevos que procesar.")
            sys.exit(0)

        total_clips = []
        for video_path in queue:
            try:
                if args.clear_cache:
                    PipelineCache(video_path, config).clear()
                clips = process_video(video_path, config, use_cache=use_cache, max_step=args.max_step)
                total_clips.extend(clips)
                if clips and move_after and args.max_step == 8:
                    mark_as_processed(video_path, processed_dir)
            except Exception as e:
                logger.error(f"Error procesando {video_path.name}: {e}")
                continue

        logger.info(f"Pipeline completado — {len(total_clips)} clip(s) generados")
        for c in total_clips:
            logger.info(f"✔ {c.name}")


if __name__ == "__main__":
    # Windows tiene un stack por defecto muy pequeño (~1MB).
    # Whisper large-v3 necesita mucho más. Lanzamos main() en un hilo con 64MB.
    threading.stack_size(64 * 1024 * 1024)
    t = threading.Thread(target=main, daemon=True)
    t.start()
    t.join()
