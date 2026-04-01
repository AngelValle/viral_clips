"""
publisher.py
Sube y programa automáticamente los clips generados a YouTube Shorts y TikTok.
"""

import json
import logging
import os
import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

def publish_to_youtube(video_path: Path, metadata_path: Path) -> bool:
    try:
        from googleapiclient.discovery import build
        from googleapiclient.http import MediaFileUpload
        from google_auth_oauthlib.flow import InstalledAppFlow
        from google.auth.transport.requests import Request
        import pickle
    except ImportError:
        logger.error("Faltan librerías de YouTube. Ejecuta: pip install google-api-python-client google-auth-oauthlib")
        return False

    if not metadata_path.exists(): return False

    with open(metadata_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    # Si en la interfaz marcaste que NO querías subirlo a YT, abortamos.
    if not meta.get("publish_yt", False):
        return False

    yt_meta = meta.get("youtube_shorts", {})
    title = yt_meta.get("titulos", ["Clip Viral"])[0][:100]
    description = yt_meta.get("descripcion", "") + "\n\n" + " ".join(meta.get("hashtags_universales",[]))
    schedule_iso = meta.get("schedule_time")

    credentials = None
    if os.path.exists("token.pickle"):
        with open("token.pickle", "rb") as token:
            credentials = pickle.load(token)

    if not credentials or not credentials.valid:
        if credentials and credentials.expired and credentials.refresh_token:
            credentials.refresh(Request())
        else:
            if not os.path.exists("client_secrets.json"):
                logger.error("Falta client_secrets.json de la API de YouTube.")
                return False
            flow = InstalledAppFlow.from_client_secrets_file("client_secrets.json", ["https://www.googleapis.com/auth/youtube.upload"])
            credentials = flow.run_local_server(port=0)
        with open("token.pickle", "wb") as token:
            pickle.dump(credentials, token)

    youtube = build("youtube", "v3", credentials=credentials)

    # ESTADO Y PROGRAMACIÓN
    status_config = {
        "privacyStatus": "private", # Para programar un vídeo en YT, DEBE subirse como privado
        "selfDeclaredMadeForKids": False
    }

    if schedule_iso:
        status_config["publishAt"] = schedule_iso
        logger.info(f"Programando YT Short para: {schedule_iso}")

    body = {
        "snippet": {
            "title": title,
            "description": description,
            "categoryId": "20", # Categoría Gaming
            "tags": meta.get("hashtags_universales",[])
        },
        "status": status_config
    }

    media = MediaFileUpload(str(video_path), chunksize=-1, resumable=True, mimetype="video/mp4")
    request = youtube.videos().insert(part="snippet,status", body=body, media_body=media)

    logger.info(f"Subiendo {video_path.name} a YouTube Shorts...")
    response = request.execute()
    logger.info(f"✅ Subido a YouTube! ID: {response.get('id')}")
    return True


def publish_to_tiktok(video_path: Path, metadata_path: Path) -> bool:
    try:
        from tiktok_uploader.upload import upload_video
    except ImportError:
        logger.error("Falta tiktok-uploader. Ejecuta: pip install tiktok-uploader playwright && playwright install")
        return False

    with open(metadata_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    # Si en la interfaz marcaste que NO querías subirlo a TikTok, abortamos.
    if not meta.get("publish_tk", False):
        return False

    tk_meta = meta.get("tiktok", {})
    description = tk_meta.get("titulos", [""])[0] + "\n" + tk_meta.get("descripcion", "")
    schedule_iso = meta.get("schedule_time")

    if not os.path.exists("cookies.txt"):
        logger.error("Falta cookies.txt para autenticarse en TikTok.")
        return False

    schedule_dt = None
    if schedule_iso:
        schedule_dt = datetime.datetime.fromisoformat(schedule_iso)
        logger.info(f"Programando TikTok para: {schedule_dt.strftime('%d/%m/%Y %H:%M')}")

    logger.info(f"Subiendo {video_path.name} a TikTok...")

    # Upload con Playwright
    failed = upload_video(
        str(video_path),
        description=description,
        schedule=schedule_dt, # Pasamos la fecha de programación a la API
        cookies="cookies.txt",
        headless=True
    )

    if failed:
        logger.error("Error al subir a TikTok.")
        return False

    logger.info("✅ Subido a TikTok con éxito.")
    return True
