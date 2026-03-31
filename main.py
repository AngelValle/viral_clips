#!/usr/bin/env python3
"""
main.py
Punto de entrada del pipeline de extracción de clips virales.

Uso:
    python main.py --file videos/input/video.mp4
    python main.py --folder videos/input/
    python main.py --file videos/input/video.mp4 --no-cache
    python main.py --file videos/input/video.mp4 --clear-cache
"""

import argparse
import json
import subprocess
import logging
import shutil
import sys
import tempfile
import warnings
import os
warnings.filterwarnings("ignore", message="Failed to launch Triton kernels")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

from pathlib import Path

from modules.scanner             import build_queue, load_config, mark_as_processed
from modules.transcriber         import transcribe, words_in_range
from modules.viral_scorer        import detect_viral_moments
from modules.exporter            import remap_words_to_fragments
from modules.face_detector       import analyze_video_faces
from modules.censor              import build_profanity_set, detect_profanity, apply_audio_censorship
from modules.subtitles           import generate_subtitles
from modules.composer            import compose_dynamic
from modules.exporter            import extract_segment, embed_subtitles, save_metadata
from modules.gpu_utils           import print_gpu_summary
from modules.metadata_generator  import generate_clip_metadata
from modules.cache               import PipelineCache

from modules.log_setup import setup_logging
setup_logging()
logger = logging.getLogger("main")


# ── Helpers ───────────────────────────────────────────────────────────────────

def get_video_duration(video_path: Path) -> float:
    import subprocess, json as _json
    result = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json",
         "-show_format", str(video_path)],
        capture_output=True, text=True,
    )
    return float(_json.loads(result.stdout)["format"]["duration"])


def clip_name_from_pattern(pattern: str, source_name: str, n: int) -> str:
    return pattern.format(source_name=source_name, n=str(n).zfill(2))


# ── Helpers de sincronía ─────────────────────────────────────────────────────

def _shift_words(words: list, delta: float) -> list:
    """
    Desplaza todos los timestamps de las palabras por `delta` segundos.
    delta negativo = adelantar (audio llega tarde → subtítulos se adelantan).
    Timestamps nunca bajan de 0.
    """
    if abs(delta) < 0.01:
        return words
    return [
        {**w,
         "start": max(0.0, round(w["start"] + delta, 3)),
         "end":   max(0.0, round(w["end"]   + delta, 3))}
        for w in words
    ]


# ── Pipeline principal ────────────────────────────────────────────────────────

def process_video(video_path: Path, config: dict, use_cache: bool = True) -> list:
    logger.info(f"Procesando: {video_path.name}")

    clips_dir = Path(config["paths"]["output_dir"]) / "clips"
    meta_dir  = Path(config["paths"]["output_dir"]) / "metadata"
    clips_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    cache = PipelineCache(video_path, config) if use_cache else None

    # ── PASO 1: Transcripción ─────────────────────────────────────────────────
    all_words = cache.get_transcription() if cache else None
    if all_words is None:
        logger.info("PASO 1/7 — Transcripción con Whisper")
        all_words = transcribe(video_path, config)
        if cache:
            cache.save_transcription(all_words)
    else:
        logger.info("PASO 1/7 — Transcripción ✓ (desde caché)")

    duration = get_video_duration(video_path)

    # ── PASO 2: Detección de momentos virales ─────────────────────────────────
    segments = cache.get_segments() if cache else None
    if segments is None:
        logger.info("PASO 2/7 — Detección de momentos virales")
        segments = detect_viral_moments(video_path, all_words, config, duration)
        if cache:
            cache.save_segments(segments)
    else:
        logger.info("PASO 2/7 — Segmentos virales ✓ (desde caché)")

    if not segments:
        logger.warning(f"No se detectaron segmentos virales en: {video_path.name}")
        return []

    # ── PASO 3–7: Procesamiento de cada segmento ──────────────────────────────
    generated_clips = []
    profanity_set   = build_profanity_set(config)
    source_name     = video_path.stem
    naming_pattern  = config["output"]["naming_pattern"]

    for idx, segment in enumerate(segments, start=1):
        logger.info(
            f"Clip {idx}/{len(segments)} "
            f"[{segment['start']:.1f}s – {segment['end']:.1f}s] "
            f"score={segment['score']:.3f}"
        )

        clip_stem  = Path(clip_name_from_pattern(naming_pattern, source_name, idx)).stem
        final_path = clips_dir / f"{clip_stem}.mp4"

        # Saltar clip si ya existe completo en disco
        if cache and cache.is_clip_done(final_path):
            logger.info(f"Clip {idx} ✓ ya existe, saltando ({final_path.name})")
            generated_clips.append(final_path)
            continue

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)

            # ── PASO 3: Extracción del segmento crudo ─────────────────────────
            logger.info("PASO 3/7 — Extrayendo segmento crudo")
            raw_clip  = tmp / "raw.mp4"
            fragments = segment.get("fragments")   # None si viene del fallback
            extract_segment(video_path, segment["start"], segment["end"],
                            raw_clip, fragments=fragments)

            # ── PASO 4: Detección facial ──────────────────────────────────────
            face_data = cache.get_face_data(clip_stem) if cache else None
            if face_data is None:
                logger.info("PASO 4/7 — Detección facial")
                face_data, is_driving = analyze_video_faces(raw_clip, config)
                if cache:
                    cache.save_face_data(clip_stem, face_data)
            else:
                logger.info("PASO 4/7 — Detección facial ✓ (desde caché)")
                # is_driving no se cachea: recalcular sobre el clip extraído
                from modules.face_detector import _is_driving_frame
                import cv2 as _cv2
                _cap   = _cv2.VideoCapture(str(raw_clip))
                _fps   = _cap.get(_cv2.CAP_PROP_FPS) or 60.0
                _every = max(1, int(_fps * 2))
                _hits, _checks, _idx = 0, 0, 0
                while True:
                    _ret, _frame = _cap.read()
                    if not _ret:
                        break
                    if _idx % _every == 0:
                        _checks += 1
                        if _is_driving_frame(_frame, config):
                            _hits += 1
                    _idx += 1
                _cap.release()
                is_driving = _checks > 0 and (_hits / _checks) >= 0.5

            # ── PASO 5: Composición 9:16 ──────────────────────────────────────
            logger.info("PASO 5/7 — Composición 9:16 dinámica")
            composed = tmp / "composed.mp4"
            compose_dynamic(raw_clip, face_data, config, composed,
                            is_driving=is_driving)

            # ── PASO 6: Censura + subtítulos + render final ───────────────────
            logger.info("PASO 6/7 — Censura, subtítulos y render final")
            # Recoger todas las palabras del rango completo del segmento
            clip_words = words_in_range(all_words, segment["start"], segment["end"])

            # Reasignar timestamps al nuevo timeline (con jump cuts si hay fragments)
            fragments  = segment.get("fragments")
            if fragments and len(fragments) > 1:
                clip_words_rel = remap_words_to_fragments(clip_words, fragments)
            else:
                # Sin fragments o fragmento único: offset simple
                offset = segment["start"]
                clip_words_rel = [
                    {**w, "start": round(w["start"] - offset, 3),
                           "end":   round(w["end"]   - offset, 3)}
                    for w in clip_words
                ]
            censored_words = detect_profanity(clip_words_rel, profanity_set)

            # Pitido: siempre timestamps originales de Whisper (relativos al clip)
            # Los asteriscos y el pitido usan exactamente los mismos timestamps.
            censored_audio = tmp / "censored.mp4"
            apply_audio_censorship(composed, censored_audio, censored_words, config)

            # Subtítulos: timestamps originales + manual_offset_sec
            sub_offset = config.get("subtitles", {}).get("manual_offset_sec", 0.0)
            sub_words  = _shift_words(censored_words, sub_offset) if sub_offset else censored_words

            # Duración real del clip compuesto (puede diferir de segment duration)
            _probe = subprocess.run(
                ["ffprobe", "-v", "quiet", "-print_format", "json",
                 "-show_format", str(censored_audio)],
                capture_output=True, text=True,
            )
            try:
                _clip_dur = float(json.loads(_probe.stdout)["format"]["duration"])
            except Exception:
                _clip_dur = segment["end"] - segment["start"]

            # Clampar end de cada palabra a clip_dur para que ningún subtítulo
            # quede fuera del rango del vídeo (causa de subtítulos que desaparecen)
            sub_words = [
                {**w,
                 "start": min(w["start"], _clip_dur),
                 "end":   min(w["end"],   _clip_dur)}
                for w in sub_words
                if w["start"] < _clip_dur   # descartar palabras fuera del clip
            ]

            ass_file = generate_subtitles(sub_words, config, tmp, clip_stem)
            embed_subtitles(censored_audio, ass_file, final_path, config)
            save_metadata(final_path, video_path, segment,
                          censored_words, config, meta_dir)

            # ── PASO 7: Metadata viral IA ─────────────────────────────────────
            if config.get("gemini", {}).get("api_key", "").strip():
                logger.info("PASO 7/7 — Generando títulos y descripciones con IA")
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
                logger.info("PASO 7/7 — Metadata IA omitida (sin API key)")

        logger.info(f"✔ Clip generado: {final_path.name}")
        generated_clips.append(final_path)

    return generated_clips


# ── Argumentos CLI ─────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Viral Clip Automation — procesa vídeos locales."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--file",   type=str, help="Ruta a un único fichero de vídeo.")
    group.add_argument("--folder", type=str, help="Ruta a carpeta de vídeos.")
    # Resolver config relativo al directorio del script, no al CWD
    script_dir = Path(__file__).parent
    default_config = str(script_dir / "config.json")
    parser.add_argument("--config",      type=str, default=default_config)
    parser.add_argument("--no-move",     action="store_true",
                        help="No mover los vídeos a 'processed' al terminar.")
    parser.add_argument("--no-cache",    action="store_true",
                        help="Ignorar caché y reprocesar todo desde cero.")
    parser.add_argument("--clear-cache", action="store_true",
                        help="Borrar la caché del vídeo antes de procesar.")
    return parser.parse_args()


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    config = load_config(args.config)
    _gkey = bool(config.get("gemini", {}).get("api_key", "").strip())
    logger.info(f"Config cargado: {args.config} | gemini key: {_gkey}")

    print_gpu_summary()

    processed_dir = Path(config["paths"]["processed_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)

    move_after = not args.no_move and config["watch_mode"].get("move_processed", True)
    use_cache  = not args.no_cache

    if args.file:
        video_path = Path(args.file)
        if args.clear_cache:
            PipelineCache(video_path, config).clear()
        clips = process_video(video_path, config, use_cache=use_cache)
        if clips and move_after:
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
                clips = process_video(video_path, config, use_cache=use_cache)
                total_clips.extend(clips)
                if move_after:
                    mark_as_processed(video_path, processed_dir)
            except Exception as e:
                logger.error(f"Error procesando {video_path.name}: {e}")
                continue

        logger.info(f"Pipeline completado — {len(total_clips)} clip(s) generados")
        for c in total_clips:
            logger.info(f"✔ {c.name}")


if __name__ == "__main__":
    main()
