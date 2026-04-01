"""
censor.py
Detecta palabras malsonantes en la transcripción y aplica:
  1. Pitido (beep 1kHz) sobre el audio en los timestamps de la palabra.
  2. Sustitución por asteriscos en los subtítulos.
Soporta perfiles de strictness por red social.
"""

import re
import logging
import shutil
import subprocess
import tempfile
import os
from pathlib import Path
from typing import List, Dict, Set, Any

logger = logging.getLogger(__name__)

# ── Lista base de palabras malsonantes (ES + EN) ───────────────────────────────
# Expande esta lista según tus necesidades. Se cargan también custom_words del config.
BASE_PROFANITY_ES = [
    "mierda", "joder", "hostia", "coño", "puta", "puto", "polla",
    "gilipollas", "imbécil", "idiota", "capullo", "cabrón", "cabron",
    "maricón", "maricon", "cojones", "follar", "follar", "pedo",
    "culo", "leche", "ostia", "ostias",
]

BASE_PROFANITY_EN = [
    "fuck", "shit", "bitch", "ass", "bastard", "crap", "damn",
    "dick", "pussy", "asshole", "motherfucker", "bullshit", "hell",
]

STRICTNESS_PROFILES = {
    "high":   BASE_PROFANITY_ES + BASE_PROFANITY_EN,
    "medium": BASE_PROFANITY_ES,
    "low":    ["mierda", "puta", "fuck", "shit"],
}


def _load_wordlist(wordlist_path: Path) -> List[str]:
    if wordlist_path.exists():
        with open(wordlist_path, "r", encoding="utf-8") as f:
            return [line.strip().lower() for line in f if line.strip()]
    return []


def build_profanity_set(config: Dict[str, Any]) -> Set[str]:
    """Construye el conjunto de palabras a censurar según perfil y config."""
    mode      = config["censorship"]["mode"]
    profile   = config["censorship"]["profiles"].get(mode, {})
    strictness = profile.get("strictness", "medium")

    words = set(STRICTNESS_PROFILES.get(strictness, []))
    words.update(w.lower() for w in config["censorship"].get("custom_words", []))

    # Cargar wordlist externa si existe
    extra = _load_wordlist(Path("assets/wordlists/custom.txt"))
    words.update(extra)

    return words


def detect_profanity(
    words: List[Dict[str, Any]],
    profanity_set: Set[str],
) -> List[Dict[str, Any]]:
    """
    Marca las palabras que deben censurarse.
    Devuelve lista con campo extra: {"censored": bool, "censored_text": str}
    """
    result = []
    for w in words:
        clean = re.sub(r"[^a-záéíóúüñA-ZÁÉÍÓÚÜÑ]", "", w["word"]).lower()
        is_bad = clean in profanity_set
        censored_text = _asterisk(w["word"]) if is_bad else w["word"]
        result.append({
            **w,
            "censored":      is_bad,
            "censored_text": censored_text,
        })
    return result


def _asterisk(word: str) -> str:
    """
    Convierte una palabra en versión censurada:
    primera letra + asteriscos (ej: 'mierda' → 'm*****').
    Palabras <= 3 letras → '***'.
    """
    clean = re.sub(r"[^a-záéíóúüñA-ZÁÉÍÓÚÜÑ]", "", word)
    if len(clean) <= 3:
        return "***"
    return clean[0] + "*" * (len(clean) - 1)


def apply_audio_censorship(input_path: Path, output_path: Path,
                            bad_words: List[Dict[str, Any]], config: Dict[str, Any]) -> Path:
    """
    Aplica el pitido al audio del vídeo usando ffmpeg.
    Si no hay palabras a censurar, copia el fichero tal cual.
    """
    beep_hz  = config["censorship"]["beep_frequency_hz"]
    targets  = [w for w in bad_words if w["censored"]]

    if not targets:
        shutil.copy2(str(input_path), str(output_path))
        logger.info("Sin palabras a censurar, audio sin cambios.")
        return output_path

    logger.info(f"Aplicando censura de audio en {len(targets)} palabra(s)...")

    # Construir filtergraph
    volume_ranges = "+".join(
        f"between(t,{w['start']},{w['end']})" for w in targets
    )
    silence_f = f"[0:a]volume=enable='{volume_ranges}':volume=0[muted]"

    # Cada pitido se genera con su duración exacta y se retrasa con adelay
    # al timestamp de inicio de la palabra (en milisegundos).
    # Así cada beep queda posicionado exactamente sobre la palabra censurada.
    beep_parts = []
    for i, w in enumerate(targets):
        dur        = max(0.05, round(w["end"] - w["start"], 3))
        delay_ms   = int(round(w["start"] * 1000))
        beep_parts.append(
            f"sine=frequency={beep_hz}:sample_rate=44100:duration={dur},"
            f"adelay={delay_ms}|{delay_ms}[b{i}]"
        )

    # Mezclar todos los pitidos posicionados con el audio silenciado
    beep_inputs = "".join(f"[b{i}]" for i in range(len(beep_parts)))
    n_mix       = len(beep_parts) + 1          # +1 por [muted]
    mix_f       = f"[muted]{beep_inputs}amix=inputs={n_mix}:normalize=0[aout]"

    filter_complex = ";".join([silence_f] + beep_parts + [mix_f])

    cmd = [
        "ffmpeg", "-y",
        "-i", str(input_path),
        "-filter_complex", filter_complex,
        "-map", "0:v",
        "-map", "[aout]",
        "-c:v", "copy",
        "-c:a", "aac",
        "-b:a", "192k",
        "-avoid_negative_ts", "make_zero",
        str(output_path),
    ]

    result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    if result.returncode != 0:
        logger.error(f"Error ffmpeg censura: {result.stderr.decode()}")
        raise RuntimeError("Fallo en la censura de audio con ffmpeg")

    logger.info(f"Audio censurado guardado: {output_path.name}")
    return output_path
