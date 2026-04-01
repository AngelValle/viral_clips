"""
transcriber.py
Transcribe el audio de un vídeo usando faster-whisper (CTranslate2) + diarización Pyannote.
"""

import gc
import logging
import os
import re
import subprocess
import tempfile
import warnings
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


def transcribe(video_path: Path, config: dict) -> List[Dict[str, Any]]:
    """
    Transcribe el vídeo con faster-whisper y devuelve palabras con timestamps.
    [{"word": str, "start": float, "end": float, "confidence": float}, ...]
    """
    from faster_whisper import WhisperModel

    model_name  = config["whisper"].get("model", "large-v3")
    language    = config["whisper"].get("language", "es")
    device_cfg  = config["whisper"].get("device", "auto")
    compute     = config["whisper"].get("compute_type", "float16")

    # Resolver device
    if device_cfg == "auto":
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"
    else:
        device = device_cfg

    # En CPU usar int8 que es más eficiente
    if device == "cpu" and compute == "float16":
        compute = "int8"

    logger.info(f"Cargando faster-whisper '{model_name}' en {device} ({compute})...")
    model = WhisperModel(model_name, device=device, compute_type=compute)

    logger.info(f"Transcribiendo: {video_path.name}")
    segments_iter, info = model.transcribe(
        str(video_path),
        language=language,
        word_timestamps=True,
        vad_filter=True,
        vad_parameters=dict(
            min_silence_duration_ms=300,   # menos agresivo → menos drift de timestamps
        ),
        beam_size=5,
        best_of=5,
        temperature=0.0,
        # Evita que el contexto anterior desplace timestamps hacia adelante.
        # Es la principal causa de drift acumulativo en vídeos largos.
        condition_on_previous_text=False,
        # Fuerza chunks de 30s para mejorar la precisión de alineación temporal.
        chunk_length=30,
        # Límite de tokens por chunk: evita "loops" de alucinación que
        # desplazan todos los timestamps siguientes.
        max_new_tokens=128,
    )

    words = []
    for segment in segments_iter:
        if segment.words is None:
            continue
        for w in segment.words:
            word_text = w.word.strip()
            if not word_text:
                continue
            words.append({
                "word":       word_text,
                "start":      round(w.start, 3),
                "end":        round(w.end,   3),
                "confidence": round(w.probability, 3),
                "speaker":    "SPEAKER_00",
            })

    logger.info(
        f"Transcripción completada: {len(words)} palabras | "
        f"idioma detectado: {info.language} ({info.language_probability:.0%})"
    )

    del model
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

    # ── Diarización de locutores (opcional, requiere HF_TOKEN en el entorno) ───
    diarization_enabled = config.get("transcriber", {}).get("diarization_enabled", True)
    hf_token = os.environ.get("HF_TOKEN", "").strip()
    if diarization_enabled and hf_token:
        try:
            # Silenciar warnings ruidosos de pyannote/torch que no son errores
            warnings.filterwarnings("ignore", message="torchcodec is not installed")
            warnings.filterwarnings("ignore", category=UserWarning, module="pyannote")
            warnings.filterwarnings("ignore", message="std\\(\\): degrees of freedom")

            logger.info("Iniciando diarización de locutores (Pyannote)...")
            from pyannote.audio import Pipeline
            import torch

            try:
                pipeline = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization-3.1", token=hf_token
                )
            except TypeError:
                pipeline = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization-3.1", use_auth_token=hf_token
                )

            if torch.cuda.is_available():
                pipeline.to(torch.device("cuda"))

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                wav_path = tmp.name

            logger.info("Extrayendo pista de audio para diarización...")
            subprocess.run(
                ["ffmpeg", "-y", "-i", str(video_path),
                 "-ar", "16000", "-ac", "1", wav_path],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )

            import wave as _wave
            import numpy as _np
            import torch as _torch
            with _wave.open(wav_path, "rb") as _wf:
                _sr     = _wf.getframerate()
                _raw    = _wf.readframes(_wf.getnframes())
            if os.path.exists(wav_path):
                os.unlink(wav_path)
            # FFmpeg extrae mono 16-bit PCM → float32 normalizado, shape (1, time)
            _audio   = _np.frombuffer(_raw, dtype=_np.int16).astype(_np.float32) / 32768.0
            waveform = _torch.from_numpy(_audio[_np.newaxis, :])
            diarization = pipeline({"waveform": waveform, "sample_rate": _sr})

            # Extraer el objeto Annotation compatible con cualquier versión de pyannote:
            #   - pyannote < 3.3   → devuelve Annotation directamente (tiene itertracks)
            #   - pyannote ≥ 3.3   → devuelve DiarizeOutput (@dataclass)
            #     con el campo .speaker_diarization (Annotation con itertracks)
            if hasattr(diarization, "itertracks"):
                # Retorno legacy: Annotation directa
                annotation = diarization
            elif hasattr(diarization, "speaker_diarization"):
                # DiarizeOutput (@dataclass pyannote ≥ 3.3)
                annotation = diarization.speaker_diarization
            else:
                # Fallback genérico: buscar cualquier atributo con itertracks
                annotation = next(
                    (v for v in vars(diarization).values()
                     if hasattr(v, "itertracks")),
                    None,
                )

            if annotation is None or not hasattr(annotation, "itertracks"):
                raise ValueError(f"No se pudo extraer Annotation de {type(diarization).__name__}")

            # Construir lista plana (start, end, speaker) para lookup eficiente
            speaker_segments = [
                (turn.start, turn.end, spk)
                for turn, _, spk in annotation.itertracks(yield_label=True)
            ]

            for w in words:
                mid = w["start"] + (w["end"] - w["start"]) / 2
                for seg_start, seg_end, spk in speaker_segments:
                    if seg_start <= mid <= seg_end:
                        w["speaker"] = spk
                        break

            logger.info("Diarización completada. Locutores asignados.")
        except Exception as exc:
            logger.warning(f"Diarización fallida ({exc}). Se usa un solo locutor.")

    return words


def words_in_range(words: List[Dict], start: float, end: float) -> List[Dict]:
    """Filtra las palabras que estan dentro del rango [start, end]."""
    return [w for w in words if w["start"] >= start and w["end"] <= end]


def _split_into_phrases(words: List[Dict], max_chars: int) -> List[List[int]]:
    """
    Agrupa los índices de palabras en frases que caben en max_chars caracteres.
    Una frase nueva empieza cuando añadir la siguiente palabra superaría max_chars.
    Devuelve lista de listas de índices: [[0,1,2,3], [4,5,6], ...]
    """
    phrases = []
    current = []
    current_len = 0

    for i, w in enumerate(words):
        text = w.get("censored_text", w["word"])
        # +1 por el espacio entre palabras
        needed = len(text) + (1 if current else 0)
        if current and current_len + needed > max_chars:
            phrases.append(current)
            current     = [i]
            current_len = len(text)
        else:
            current.append(i)
            current_len += needed

    if current:
        phrases.append(current)

    return phrases


def build_highlighted_entries(
    words: List[Dict],
    max_line_width: int = 20,
    max_line_count: int = 1,
) -> List[Dict[str, Any]]:
    """
    Genera subtítulos estilo karaoke con frase completa fija:
      - Las palabras se agrupan en frases que caben en max_line_width caracteres.
      - La frase entera es visible durante toda su duración.
      - La palabra activa se marca con <u>...</u> y avanza de izquierda a derecha.
      - Cuando la frase termina, aparece la siguiente frase completa.

    Ejemplo con max_line_width=20:
      Frase: "hola como estás bien"
        t=0.0: "<u>hola</u> como estás bien"
        t=0.3: "hola <u>como</u> estás bien"
        t=0.7: "hola como <u>estás</u> bien"
        t=1.1: "hola como estás <u>bien</u>"
      Frase: "qué tal todo por"
        t=1.5: "<u>qué</u> tal todo por"
        ...

    Devuelve:
    [{"start": float, "end": float, "text": str}, ...]
    """
    if not words:
        return []

    phrases  = _split_into_phrases(words, max_line_width)
    entries  = []

    for phrase_indices in phrases:
        main_speaker = words[phrase_indices[0]].get("speaker", "SPEAKER_00")

        for active_idx in phrase_indices:
            active_word = words[active_idx]

            parts = []
            for idx in phrase_indices:
                text = words[idx].get("censored_text", words[idx]["word"])
                if idx == active_idx:
                    parts.append(f"<u>{text}</u>")
                else:
                    parts.append(text)

            entries.append({
                "start":   active_word["start"],
                "end":     active_word["end"],
                "text":    " ".join(parts).strip(),
                "speaker": main_speaker,
            })

    return entries


# ── Generación de subtítulos ASS ─────────────────────────────────────────────

# Paleta de colores BGR Hex para ASS (Blanco, Amarillo, Verde claro, Cian, Rosa)
SPEAKER_COLORS = ["&H00FFFFFF", "&H0000FFFF", "&H00B2FF66", "&H00FFFF00", "&H00FF99FF"]


def _ass_time(s: float) -> str:
    h, m, sec = int(s // 3600), int((s % 3600) // 60), s % 60
    return f"{h}:{m:02d}:{sec:05.2f}"


def _convert_highlight_tags(text: str, base_color: str) -> str:
    """La palabra activa se renderiza en rojo; el resto usa el color del locutor."""
    text = re.sub(
        r"<u>(.*?)</u>",
        lambda m: f"{{\\c&H0000FF&}}{m.group(1)}{{\\c{base_color}}}",
        text,
    )
    return re.sub(r"<[^>]+>", "", text)


def write_ass(entries: List[Dict[str, Any]], output_path: Path, config: Dict[str, Any]) -> Path:
    """Genera un archivo .ass con estilo TikTok y colores por locutor."""
    cfg_s  = config["subtitles"]
    cfg_o  = config["output"]
    res_w, res_h  = cfg_o["resolution_w"], cfg_o["resolution_h"]
    font_size     = cfg_s["font_size"]
    outline_w     = cfg_s["outline_width"]
    pos_y         = int(res_h * cfg_s["position_y_ratio"])
    font_name     = cfg_s.get("font_name", "Showcard Gothic")

    header = (
        f"[Script Info]\nScriptType: v4.00+\n"
        f"PlayResX: {res_w}\nPlayResY: {res_h}\nWrapStyle: 1\n\n"
        f"[V4+ Styles]\n"
        f"Format: Name, Fontname, Fontsize, PrimaryColour, OutlineColour, BackColour, Bold, Italic, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV\n"
        f"Style: TikTok,{font_name},{font_size},&H00FFFFFF,&H00000000,&H00000000,1,0,1,{outline_w},0,2,60,60,60\n\n"
        f"[Events]\n"
        f"Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    )

    speaker_map: Dict[str, str] = {}
    color_idx = 0
    event_lines = []

    for e in entries:
        spk = e.get("speaker", "SPEAKER_00")
        if spk not in speaker_map:
            speaker_map[spk] = SPEAKER_COLORS[color_idx % len(SPEAKER_COLORS)]
            color_idx += 1

        base_color = speaker_map[spk]
        text       = _convert_highlight_tags(e["text"], base_color)
        event_lines.append(
            f"Dialogue: 0,{_ass_time(e['start'])},{_ass_time(e['end'])},"
            f"TikTok,,0,0,0,,"
            f"{{\\an2\\pos({res_w // 2},{pos_y})}}{{\\c{base_color}}}{text}"
        )

    output_path.write_text(header + "\n".join(event_lines) + "\n", encoding="utf-8")
    return output_path


def generate_subtitles(
    words: List[Dict[str, Any]],
    config: Dict[str, Any],
    output_dir: Path,
    clip_name: str,
) -> Path:
    """Orquesta build_highlighted_entries → write_ass y devuelve la ruta del .ass."""
    entries = build_highlighted_entries(
        words,
        max_line_width=config["subtitles"].get("max_line_width", 20),
        max_line_count=config["subtitles"].get("max_line_count", 1),
    )
    return write_ass(entries, output_dir / f"{clip_name}.ass", config)
