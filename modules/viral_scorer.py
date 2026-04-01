"""
viral_scorer.py
Detecta los momentos más virales de un vídeo enviando la transcripción
completa a Gemini para que identifique directamente los segmentos virales.

Si no hay API key de Gemini configurada, cae back a detección por señales
cuantitativas (audio RMS + velocidad de habla + keywords).
"""

import json
import logging
import os
import re
import subprocess
import tempfile
import time
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional

from modules.gpu_utils import get_torch_device, free_gpu_memory

logger = logging.getLogger(__name__)


# ── Gemini (google-genai SDK, clave desde variable de entorno) ────────────────

def _get_safety_settings():
    """Deshabilita todos los filtros de contenido de Gemini."""
    from google.genai import types as genai_types
    categories = [
        "HARM_CATEGORY_HARASSMENT",
        "HARM_CATEGORY_HATE_SPEECH",
        "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "HARM_CATEGORY_DANGEROUS_CONTENT",
    ]
    return [
        genai_types.SafetySetting(category=c, threshold="BLOCK_NONE")
        for c in categories
    ]


def _extract_frames(
    video_path: Path,
    video_duration: float,
    tmp_dir: Path,
    interval_sec: float = 10.0,
) -> List[tuple]:
    """
    Extrae 1 frame cada `interval_sec` segundos con FFmpeg.
    JPEG 480px ancho, calidad baja (~15-30 KB c/u) para minimizar tokens.
    Compatible con el tier gratuito de Gemini (sin Files API).
    Devuelve [(timestamp_float, Path), ...].
    """
    frames = []
    t = interval_sec / 2.0
    while t < video_duration:
        out = tmp_dir / f"frame_{int(t):06d}s.jpg"
        r = subprocess.run(
            ["ffmpeg", "-y", "-ss", str(t), "-i", str(video_path),
             "-frames:v", "1", "-vf", "scale=480:-2", "-q:v", "8", str(out)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        if r.returncode == 0 and out.exists():
            frames.append((t, out))
        t += interval_sec
    return frames


def _call_gemini(
    prompt: str,
    config: dict,
    max_tokens: int = 4096,
    frames: Optional[List[tuple]] = None,
    max_retries: int = 50,
    retry_wait: float = 20.0,
) -> Optional[str]:
    """
    Llama a Gemini con reintentos para el tier gratuito (rate limit 15 RPM).
    Si se pasan `frames`, construye una request multimodal con imágenes inline
    (sin Files API — compatible con el tier gratuito).
    """
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        logger.warning("GEMINI_API_KEY no encontrada en el entorno")
        return None

    model = config.get("gemini", {}).get("model", "gemini-3.1-flash-lite-preview")

    from google import genai
    from google.genai import types as genai_types

    client = genai.Client(api_key=api_key)

    if frames:
        # Multimodal: frames como bytes inline + timestamp + prompt al final
        parts = []
        for t, img_path in frames:
            parts.append(genai_types.Part.from_bytes(
                data=img_path.read_bytes(), mime_type="image/jpeg",
            ))
            parts.append(genai_types.Part(text=f"[Frame en t={t:.0f}s]"))
        parts.append(genai_types.Part(text=prompt))
        contents = [genai_types.Content(parts=parts, role="user")]
    else:
        contents = [prompt]

    for attempt in range(1, max_retries + 1):
        try:
            response = client.models.generate_content(
                model=model,
                contents=contents,
                config=genai_types.GenerateContentConfig(
                    temperature=0.2,
                    max_output_tokens=max_tokens,
                    safety_settings=_get_safety_settings(),
                    response_mime_type="application/json",
                ),
            )
            return response.text
        except Exception as exc:
            err = str(exc)
            is_transient = any(k in err for k in (
                "429", "quota", "RESOURCE_EXHAUSTED", "rate",
                "503", "UNAVAILABLE", "high demand",
                "500", "INTERNAL", "overloaded",
            ))
            if is_transient and attempt < max_retries:
                logger.warning(
                    f"Gemini API error transitorio (intento {attempt}/{max_retries}): {exc}. "
                    f"Reintentando en {retry_wait:.0f}s..."
                )
                time.sleep(retry_wait)
            else:
                logger.warning(f"Gemini API error (intento {attempt}): {exc}")
                return None
    return None


# ── Prompt principal ──────────────────────────────────────────────────────────

def _build_transcript_text(words: List[Dict]) -> str:
    """Agrupa palabras en bloques legibles con timestamps para Gemini."""
    lines   = []
    block   = []
    block_t = None
    for i, w in enumerate(words):
        if block_t is None:
            block_t = w["start"]
        block.append(w["word"])
        next_w = words[i + 1] if i + 1 < len(words) else None
        if (next_w is None
                or (next_w["start"] - w["end"]) > 1.5
                or (w["end"] - block_t) > 10):
            lines.append(f"[{block_t:.1f}s] {' '.join(block)}")
            block   = []
            block_t = None
    return "\n".join(lines)


def _build_viral_prompt(words: List[Dict], config: dict,
                        video_duration: float) -> str:
    cfg_v    = config["viral_detection"]
    min_dur  = cfg_v["min_clip_duration"]
    max_dur  = cfg_v["max_clip_duration"]
    top_n    = cfg_v.get("top_n_clips", 999)
    streamer = config.get("claude", {}).get("streamer_name", "")
    game     = config.get("claude", {}).get("game_name", "GTA V")
    keywords = cfg_v.get("viral_keywords", [])
    min_frag = cfg_v.get("min_fragment_dur", 1.5)

    transcript_text = _build_transcript_text(words)
    kw_hint    = f"\nMomentos clave para este streamer: {', '.join(keywords)}" if keywords else ""
    limit_hint = f"Detecta hasta {top_n} clips." if top_n < 999 else "Detecta todos los clips que merezcan la pena."

    return f"""Eres un editor experto en clips virales de streaming hispanohablante con estilo TikTok.

Tienes la transcripción completa de un directo de {game} del streamer {streamer}.
Duración total: {video_duration:.0f}s ({video_duration/60:.1f} min).{kw_hint}

Tu tarea es crear clips virales como haría un editor profesional de TikTok:
- Cada clip se compone de VARIOS FRAGMENTOS del directo unidos con jump cuts.
- Los fragmentos pueden venir de cualquier parte del directo, no tienen que ser contiguos.
- Se eliminan silencios, esperas, momentos sin gracia o relleno entre los fragmentos.
- El resultado es un clip denso, sin momentos muertos, con alta densidad de contenido.

CRITERIOS PARA FRAGMENTOS A INCLUIR:
- Reacciones emocionales intensas (sorpresa, risa, enfado, gritos)
- Frases graciosas, ingeniosas o impactantes
- Momentos de tensión con resolución clara
- Conversaciones divertidas o situaciones inesperadas
- Cada fragmento debe empezar y terminar en pausa natural del habla

CRITERIOS PARA EXCLUIR (no incluir estos fragmentos):
- Silencios de más de 1 segundo sin reacción
- Momentos de espera, loading, o gameplay sin comentario
- Transiciones sin contenido relevante
- Frases incompletas sin resolución

RESTRICCIONES:
- Duración TOTAL del clip (suma de fragmentos): mínimo {min_dur}s, máximo {max_dur}s
- Duración mínima por fragmento individual: {min_frag}s
- Cada fragmento: start/end en pausa natural del habla (no cortar palabras)
- Los fragmentos dentro de un clip deben estar ordenados cronológicamente
- No solapar fragmentos entre distintos clips
- {limit_hint}

TRANSCRIPCIÓN (formato [timestamp_inicio] texto):
{transcript_text}

Responde ÚNICAMENTE con un array JSON válido, sin texto adicional ni bloques de código.
Estructura exacta:
[
  {{
    "score": FLOAT 0.0-1.0,
    "descripcion": "qué ocurre en este clip en una frase",
    "fragments": [
      {{"start": FLOAT, "end": FLOAT}},
      {{"start": FLOAT, "end": FLOAT}}
    ]
  }},
  ...
]

Cada "fragments" debe tener al menos 1 elemento. Usa timestamps exactos de la transcripción.
score: 1.0 = momento excepcional, 0.5 = interesante, 0.2 = aceptable (incluir si hay algo de contenido).
Sé generoso: es mejor incluir un clip mediocre que perderse uno bueno. Mínimo score para incluir: 0.2.
Si no hay ningún momento con score ≥ 0.2 responde: []"""


# ── Parser de respuesta Gemini ────────────────────────────────────────────────

def _clean_json_response(response: str) -> str:
    """Extrae JSON limpio de la respuesta de Gemini."""
    text = response.strip()
    if "```" in text:
        for part in text.split("```"):
            part = part.strip()
            if part.startswith("json"):
                part = part[4:].strip()
            if part.startswith("["):
                return part
    m = re.search(r'\[.*\]', text, re.DOTALL)
    return m.group() if m else ""


def _parse_gemini_segments(response: str, words: List[Dict],
                            config: dict, valid_start: float,
                            valid_end: float) -> List[Dict]:
    cfg_v    = config["viral_detection"]
    min_dur  = cfg_v["min_clip_duration"]
    max_dur  = cfg_v["max_clip_duration"]
    pre_buf  = cfg_v["pre_buffer_seconds"]
    post_buf = cfg_v["post_buffer_seconds"]
    min_frag = cfg_v.get("min_fragment_dur", 1.5)

    raw_json = _clean_json_response(response)
    if not raw_json:
        logger.warning("Gemini no devolvió un array JSON válido")
        return []

    try:
        raw_clips = json.loads(raw_json)
    except json.JSONDecodeError as e:
        logger.warning(f"Error parseando JSON de Gemini: {e}")
        return []

    if not isinstance(raw_clips, list):
        return []

    segments = []
    used     = []   # lista de (start, end) de fragmentos ya asignados

    for item in raw_clips:
        try:
            score = float(item.get("score", 0.5))
            desc  = item.get("descripcion", "")
            raw_frags = item.get("fragments", [])

            # Compatibilidad: si Gemini devuelve formato antiguo (start/end planos)
            if not raw_frags and "start" in item and "end" in item:
                raw_frags = [{"start": item["start"], "end": item["end"]}]
        except (KeyError, ValueError, TypeError):
            continue

        # Validar y limpiar fragmentos (ordenados por inicio para detección de solapamiento)
        raw_frags_sorted = sorted(raw_frags, key=lambda f: float(f.get("start", 0)))
        valid_frags = []
        for frag in raw_frags_sorted:
            try:
                fs = max(valid_start, float(frag["start"]) - pre_buf)
                fe = min(valid_end,   float(frag["end"])   + post_buf)
            except (KeyError, ValueError, TypeError):
                continue

            if fe - fs < min_frag:
                logger.debug(f"Fragmento {fs:.1f}-{fe:.1f}s demasiado corto, omitido")
                continue

            # Solapamiento con fragmentos ya usados en otros clips
            if any(not (fe <= u[0] or fs >= u[1]) for u in used):
                logger.debug(f"Fragmento {fs:.1f}-{fe:.1f}s solapado con otro clip, omitido")
                continue

            # Solapamiento con fragmentos del mismo clip (evita contenido repetido)
            if any(not (fe <= vf[0] or fs >= vf[1]) for vf in valid_frags):
                logger.debug(f"Fragmento {fs:.1f}-{fe:.1f}s solapado dentro del mismo clip, omitido")
                continue

            valid_frags.append((round(fs, 2), round(fe, 2)))

        if not valid_frags:
            continue

        # Validar duración total del clip
        total_dur = sum(e - s for s, e in valid_frags)
        if total_dur < min_dur:
            logger.debug(f"Clip con {len(valid_frags)} frags dura {total_dur:.1f}s < {min_dur}s, omitido")
            continue
        if total_dur > max_dur:
            # Truncar fragmentos desde el final hasta cumplir max_dur
            trimmed = []
            acc = 0.0
            for fs, fe in valid_frags:
                dur = fe - fs
                if acc + dur <= max_dur:
                    trimmed.append((fs, fe))
                    acc += dur
                else:
                    trimmed.append((fs, round(fs + max_dur - acc, 2)))
                    break
            valid_frags = trimmed
            total_dur   = sum(e - s for s, e in valid_frags)

        # Registrar fragmentos como usados
        for fs, fe in valid_frags:
            used.append((fs, fe))

        # start/end del clip = rango completo de sus fragmentos (para caché y logs)
        clip_start = valid_frags[0][0]
        clip_end   = valid_frags[-1][1]

        segments.append({
            "start":       clip_start,
            "end":         clip_end,
            "score":       round(min(1.0, max(0.0, score)), 4),
            "gemini_desc": desc,
            "fragments":   [{"start": s, "end": e} for s, e in valid_frags],
        })

        frag_str = " + ".join(f"{s:.1f}-{e:.1f}s" for s, e in valid_frags)
        logger.info(f"  ✓ [{frag_str}] total={total_dur:.1f}s score={score:.2f} — {desc}")

    return segments


# ── Fallback: señales cuantitativas ──────────────────────────────────────────
# Usado solo si no hay API key de Gemini

def _extract_audio_array(video_path: Path, sample_rate: int = 16000) -> np.ndarray:
    import tempfile, os, wave
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(video_path),
             "-ar", str(sample_rate), "-ac", "1", "-f", "wav", tmp_path],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True
        )
        with wave.open(tmp_path, "rb") as wf:
            frames = wf.readframes(wf.getnframes())
            return np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    finally:
        os.unlink(tmp_path)


_DEFAULT_SCORE_WEIGHTS = {"audio_peak": 0.45, "speech_speed": 0.35, "keywords": 0.2}


def _detect_by_signals(video_path: Path, words: List[Dict], config: dict,
                        video_duration: float,
                        valid_start: float, valid_end: float) -> List[Dict]:
    """Detección legacy por señales cuantitativas (fallback sin Gemini)."""
    cfg_v    = config["viral_detection"]
    weights  = cfg_v.get("score_weights", _DEFAULT_SCORE_WEIGHTS)
    min_dur  = cfg_v["min_clip_duration"]
    max_dur  = cfg_v["max_clip_duration"]
    pre_buf  = cfg_v["pre_buffer_seconds"]
    post_buf = cfg_v["post_buffer_seconds"]
    top_n    = cfg_v["top_n_clips"]

    sample_rate = 16000
    audio = _extract_audio_array(video_path, sample_rate)
    window = int(1.0 * sample_rate)

    # RMS
    device = get_torch_device()
    try:
        import torch
        t = torch.tensor(audio, dtype=torch.float32, device=device)
        n = len(t) // window
        rms = t[:n*window].reshape(n, window).pow(2).mean(dim=1).sqrt().cpu().numpy()
    except Exception:
        n = len(audio) // window
        rms = np.sqrt((audio[:n*window].reshape(n, window) ** 2).mean(axis=1))

    rms_norm = rms / rms.max() if rms.max() > 0 else rms

    # Speech speed
    if words:
        end_t  = video_duration
        starts = np.array([w["start"] for w in words])
        ts     = np.arange(0.0, end_t, 1.0)
        spd    = (np.searchsorted(starts, ts + 5.0) - np.searchsorted(starts, ts)).astype(float)
        spd   /= spd.max() if spd.max() > 0 else 1.0
    else:
        ts, spd = np.array([]), np.array([])

    # Keywords
    kws = set(k.lower() for k in cfg_v.get("viral_keywords", []))
    if words and kws:
        kw_ts  = np.array([w["start"] for w in words if w["word"].lower() in kws])
        if len(kw_ts):
            kw_s = (np.searchsorted(kw_ts, ts + 5.0) - np.searchsorted(kw_ts, ts)).astype(float)
            kw_s /= kw_s.max() if kw_s.max() > 0 else 1.0
        else:
            kw_s = np.zeros_like(ts)
    else:
        kw_s = np.zeros_like(ts)

    scores: Dict[float, float] = {}
    for i, t in enumerate(ts):
        if t < valid_start or t > valid_end:
            continue
        rms_v = float(rms_norm[int(t)]) if int(t) < len(rms_norm) else 0.0
        spd_v = float(spd[i])           if i < len(spd)           else 0.0
        kw_v  = float(kw_s[i])          if i < len(kw_s)          else 0.0
        scores[t] = (weights["audio_peak"]   * rms_v +
                     weights["speech_speed"] * spd_v +
                     weights["keywords"]     * kw_v)

    segments = []
    used     = []
    for ts_val, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
        if len(segments) >= top_n:
            break
        start = max(valid_start, ts_val - pre_buf)
        end   = min(valid_end, ts_val + max_dur - pre_buf + post_buf)
        end   = min(end, start + max_dur)
        if end - start < min_dur:
            end = min(video_duration, start + min_dur)
        if any(not (end <= u[0] or start >= u[1]) for u in used):
            continue
        segments.append({"start": round(start, 2), "end": round(end, 2),
                         "score": round(score, 4)})
        used.append((start, end))

    return segments


# ── Punto de entrada ──────────────────────────────────────────────────────────

def detect_viral_moments(
    video_path: Path,
    words: List[Dict],
    config: dict,
    video_duration: float,
) -> List[Dict[str, Any]]:
    cfg_v       = config["viral_detection"]
    skip_intro  = cfg_v.get("skip_intro_sec", 0)
    skip_outro  = cfg_v.get("skip_outro_sec", 0)
    valid_start = float(skip_intro)
    valid_end   = video_duration - float(skip_outro)
    has_gemini  = bool(os.environ.get("GEMINI_API_KEY", "").strip())

    logger.info(f"Zona válida: {valid_start:.0f}s – {valid_end:.0f}s "
                f"(excluyendo {skip_intro}s intro + {skip_outro}s outro)")

    # Filtrar palabras fuera de la zona válida
    valid_words = [w for w in words
                   if w["start"] >= valid_start and w["end"] <= valid_end]

    if not valid_words:
        logger.warning("Sin palabras en zona válida — no se puede detectar momentos virales")
        return []

    if has_gemini:
        multimodal = config.get("ai_features", {}).get("multimodal_video", False)
        frames     = []
        _tmp_obj   = None

        if multimodal:
            interval_sec = float(config.get("ai_features", {}).get("multimodal_interval_sec", 10))
            _tmp_obj = tempfile.mkdtemp()
            tmp_path = Path(_tmp_obj)
            logger.info(f"Multimodal: extrayendo frames del vídeo (1 cada {interval_sec:.0f}s)...")
            frames = _extract_frames(video_path, video_duration, tmp_path, interval_sec=interval_sec)
            logger.info(f"Multimodal: {len(frames)} frames extraídos para análisis visual")

        logger.info(f"Detección viral con Gemini{'  + visión' if frames else ''} "
                    f"({len(valid_words)} palabras, {video_duration/60:.1f} min)...")

        prompt   = _build_viral_prompt(valid_words, config, video_duration)
        response = _call_gemini(prompt, config, max_tokens=16384,
                                frames=frames if frames else None)

        if _tmp_obj:
            import shutil as _shutil
            _shutil.rmtree(_tmp_obj, ignore_errors=True)

        if response is not None:
            segments = _parse_gemini_segments(
                response, valid_words, config, valid_start, valid_end
            )
            if segments:
                segments.sort(key=lambda s: s["start"])
                logger.info(f"Segmentos virales detectados por Gemini: {len(segments)}")
                free_gpu_memory()
                return segments
            else:
                logger.warning("Gemini no devolvió segmentos válidos — "
                               "usando detección por señales como fallback")
        else:
            logger.warning("Gemini no disponible — "
                           "usando detección por señales como fallback")
    else:
        logger.info("Sin API key Gemini — detección por señales cuantitativas")

    # Fallback: señales cuantitativas
    segments = _detect_by_signals(
        video_path, valid_words, config, video_duration, valid_start, valid_end
    )
    segments.sort(key=lambda s: s["start"])
    logger.info(f"Segmentos virales por señales: {len(segments)}")
    free_gpu_memory()
    return segments
