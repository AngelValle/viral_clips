"""
transcriber.py
Transcribe el audio de un vídeo usando faster-whisper (CTranslate2).
Mismo motor que Purfview's Faster-Whisper-XXL, accesible como librería Python.

Ventajas frente a openai-whisper:
  - 2-4x más rápido en GPU
  - Menor uso de VRAM (cuantización int8/float16)
  - Mismo modelo large-v3

Instalación (si no está ya):
  pip install faster-whisper --break-system-packages
"""

import logging
from pathlib import Path
from typing import List, Dict, Any

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
            })

    logger.info(
        f"Transcripción completada: {len(words)} palabras | "
        f"idioma detectado: {info.language} ({info.language_probability:.0%})"
    )

    del model
    try:
        import torch, gc
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    except Exception:
        pass

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
        # Construir el texto base de la frase (sin highlight)
        phrase_words = [words[i] for i in phrase_indices]

        for active_pos, active_idx in enumerate(phrase_indices):
            active_word = words[active_idx]

            # Construir la línea con highlight solo en la palabra activa
            parts = []
            for pos, idx in enumerate(phrase_indices):
                text = words[idx].get("censored_text", words[idx]["word"])
                if idx == active_idx:
                    parts.append(f"<u>{text}</u>")
                else:
                    parts.append(text)

            line = " ".join(parts).strip()
            entries.append({
                "start": active_word["start"],
                "end":   active_word["end"],
                "text":  line,
            })

    return entries
