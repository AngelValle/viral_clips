"""
metadata_generator.py
Genera títulos y descripciones optimizados para TikTok y YouTube Shorts
usando la API de Gemini (Google), basándose en el transcript del clip.

Para cada clip genera:
  - 3 títulos para TikTok  (máx 80 chars, gancho fuerte, emojis)
  - 3 títulos para YouTube Shorts (máx 70 chars, SEO)
  - Descripción TikTok     (máx 220 chars, hashtags, CTA)
  - Descripción YouTube    (máx 400 chars, hashtags SEO)
  - 10 hashtags universales

Configuración en config.json:
  gemini.api_key    : clave de la API de Google AI Studio
  gemini.model      : modelo a usar (default: gemini-2.0-flash)
  claude.streamer_name, claude.game_name, claude.content_type  (reutilizados)
"""

import json
import logging
import urllib.request
import urllib.error
from pathlib import Path
from typing import Dict, Any

logger = logging.getLogger(__name__)

GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
GEMINI_MODEL   = "gemini-2.0-flash"


def _call_gemini(prompt: str, config: dict) -> str:
    """
    Llama a la API de Gemini con retry exponencial.
    Requiere config["gemini"]["api_key"].
    """
    import time

    api_key = config.get("gemini", {}).get("api_key", "")
    if not api_key:
        raise ValueError(
            "Falta config['gemini']['api_key']. "
            "Añade tu API key de Google AI Studio en config.json."
        )

    model = config.get("gemini", {}).get("model", GEMINI_MODEL)
    url   = GEMINI_API_URL.format(model=model) + f"?key={api_key}"

    payload = json.dumps({
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature":     0.7,
            "maxOutputTokens": 1024,
        },
    }).encode("utf-8")

    max_retries = 4
    for attempt in range(max_retries):
        try:
            import socket
            req = urllib.request.Request(
                url, data=payload,
                headers={"Content-Type": "application/json"}, method="POST",
            )
            # socket.setdefaulttimeout garantiza timeout en lectura en Windows
            old_timeout = socket.getdefaulttimeout()
            socket.setdefaulttimeout(45)
            try:
                with urllib.request.urlopen(req) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
            finally:
                socket.setdefaulttimeout(old_timeout)
            return data["candidates"][0]["content"]["parts"][0]["text"]
        except urllib.error.HTTPError as e:
            if e.code == 429:
                wait = 10
                logger.warning(f"Gemini rate limit (429) — reintentando en {wait}s "
                               f"(intento {attempt+1}/{max_retries})")
                time.sleep(wait)
            else:
                body = e.read().decode("utf-8", errors="ignore")
                raise RuntimeError(f"Error API Gemini {e.code}: {body[:300]}")
        except OSError:
            wait = 10
            logger.warning(f"Gemini timeout — reintentando en {wait}s "
                           f"(intento {attempt+1}/{max_retries})")
            time.sleep(wait)

    raise RuntimeError("Gemini: máximo de reintentos alcanzado")


def _build_prompt(transcript: str, viral_score: float,
                  duration: float, config: dict) -> str:
    streamer = config.get("claude", {}).get("streamer_name", "")
    game     = config.get("claude", {}).get("game_name", "")

    context_parts = []
    if streamer:
        context_parts.append(f"Streamer: {streamer}")
    if game:
        context_parts.append(f"Juego: {game}")
    context_str = " | ".join(context_parts) if context_parts else "contenido de gaming"

    transcript_short = transcript[:600] if len(transcript) > 600 else transcript

    return f"""Eres un experto en marketing de contenido gaming para redes sociales hispanohablantes.
Analiza este clip de directo y genera metadata optimizada para maximizar el alcance viral.

CONTEXTO: {context_str}
DURACIÓN DEL CLIP: {duration:.0f} segundos
SCORE VIRAL (0-1): {viral_score:.2f}
TRANSCRIPT DEL CLIP:
\"\"\"{transcript_short}\"\"\"

Genera la siguiente metadata en ESPAÑOL. Responde ÚNICAMENTE con JSON válido, sin texto adicional, sin bloques de código:

{{
  "tiktok": {{
    "titulos": [
      "título 1 (máx 80 chars, gancho emocional fuerte, puede incluir 1-2 emojis relevantes)",
      "título 2 (máx 80 chars, genera intriga o FOMO)",
      "título 3 (máx 80 chars, variante con pregunta o dato impactante)"
    ],
    "descripcion": "descripción máx 220 chars con CTA natural y 3-5 hashtags al final. Tono conversacional, sin forzar."
  }},
  "youtube_shorts": {{
    "titulos": [
      "título 1 (máx 70 chars, incluye palabra clave del juego o momento, sin emojis)",
      "título 2 (máx 70 chars, SEO con términos de búsqueda reales)",
      "título 3 (máx 70 chars, variante descriptiva del momento)"
    ],
    "descripcion": "descripción máx 400 chars orientada a SEO: describe el momento, incluye el nombre del juego, canal y 5-8 hashtags relevantes al final."
  }},
  "hashtags_universales": [
    "#hashtag1", "#hashtag2", "#hashtag3", "#hashtag4", "#hashtag5",
    "#hashtag6", "#hashtag7", "#hashtag8", "#hashtag9", "#hashtag10"
  ]
}}"""


def generate_clip_metadata(
    transcript: str,
    viral_score: float,
    duration: float,
    clip_name: str,
    config: dict,
    output_dir: Path,
) -> Dict[str, Any]:
    """
    Genera y guarda metadata de marketing para un clip.
    Devuelve el dict con toda la metadata generada.
    """
    logger.info(f"Generando metadata viral para: {clip_name}")

    try:
        prompt   = _build_prompt(transcript, viral_score, duration, config)
        response = _call_gemini(prompt, config)

        # Limpiar posibles bloques de código en la respuesta
        response = response.strip()
        if response.startswith("```"):
            response = response.split("```")[1]
            if response.startswith("json"):
                response = response[4:]
        response = response.strip()

        metadata = json.loads(response)

    except json.JSONDecodeError as e:
        logger.warning(f"Respuesta JSON inválida de Gemini: {e}. Usando metadata genérica.")
        metadata = _fallback_metadata(clip_name, config)
    except Exception as e:
        logger.warning(f"Error generando metadata con Gemini: {e}. Usando metadata genérica.")
        metadata = _fallback_metadata(clip_name, config)

    metadata["clip_name"]    = clip_name
    metadata["viral_score"]  = viral_score
    metadata["duration_sec"] = duration

    meta_path = output_dir / f"{clip_name}_metadata.json"
    meta_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    _write_readable_txt(metadata, output_dir / f"{clip_name}_metadata.txt")

    logger.info(f"Metadata guardada: {meta_path.name}")
    return metadata


def _fallback_metadata(clip_name: str, config: dict) -> Dict[str, Any]:
    """Metadata genérica si la API falla."""
    game    = config.get("claude", {}).get("game_name", "")
    game_ht = f"#{game.replace(' ','')}" if game else "#gaming"
    return {
        "tiktok": {
            "titulos": [
                "Momento épico en el directo 🔥",
                "No te puedes perder esto",
                "¿Qué harías tú en esta situación?",
            ],
            "descripcion": f"Sígueme para más clips 👀 {game_ht} #clips #directo #gaming",
        },
        "youtube_shorts": {
            "titulos": [
                f"Momento increíble en directo {game}".strip(),
                f"Clip viral del directo {game}".strip(),
                f"Lo mejor del directo - {clip_name}",
            ],
            "descripcion": (
                f"Clip del directo. {game} gameplay en español. "
                f"Suscríbete para más contenido. "
                f"{game_ht} #gaming #clips #directo #shorts"
            ),
        },
        "hashtags_universales": [
            "#gaming", "#clips", "#directo", "#twitch", "#streamer",
            "#viral", "#shorts", "#tiktokgaming", game_ht, "#español",
        ],
    }


def _write_readable_txt(metadata: Dict, path: Path) -> None:
    """Genera un .txt legible con toda la metadata."""
    lines = [
        f"═══ METADATA: {metadata.get('clip_name', '')} ═══",
        f"Score viral: {metadata.get('viral_score', 0):.2f} | "
        f"Duración: {metadata.get('duration_sec', 0):.0f}s",
        "",
        "── TIKTOK ──────────────────────────────",
        "Títulos:",
    ]
    for i, t in enumerate(metadata.get("tiktok", {}).get("titulos", []), 1):
        lines.append(f"  {i}. {t}")
    lines += [
        "",
        "Descripción:",
        f"  {metadata.get('tiktok', {}).get('descripcion', '')}",
        "",
        "── YOUTUBE SHORTS ──────────────────────",
        "Títulos:",
    ]
    for i, t in enumerate(metadata.get("youtube_shorts", {}).get("titulos", []), 1):
        lines.append(f"  {i}. {t}")
    lines += [
        "",
        "Descripción:",
        f"  {metadata.get('youtube_shorts', {}).get('descripcion', '')}",
        "",
        "── HASHTAGS UNIVERSALES ────────────────",
        "  " + " ".join(metadata.get("hashtags_universales", [])),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
