"""
cache.py
Sistema de caché para el pipeline de clips virales.
"""

import hashlib
import json
import logging
from pathlib import Path
from typing import Optional, List, Dict

logger = logging.getLogger(__name__)


def _video_fingerprint(video_path: Path) -> str:
    stat = video_path.stat()
    return hashlib.md5(f"{stat.st_size}:{stat.st_mtime}".encode()).hexdigest()[:12]


def _config_fingerprint(config: dict, keys: List[str]) -> str:
    subset = {}
    for key in keys:
        parts = key.split(".")
        val   = config
        for p in parts:
            val = val.get(p, {}) if isinstance(val, dict) else {}
        subset[key] = val
    return hashlib.md5(json.dumps(subset, sort_keys=True).encode()).hexdigest()[:8]


class PipelineCache:

    TRANSCRIPTION_CFG_KEYS = ["whisper.model", "whisper.language", "whisper.device"]
    SEGMENTS_CFG_KEYS      = [
        "viral_detection.min_clip_duration",
        "viral_detection.max_clip_duration",
        "viral_detection.top_n_clips",
        "viral_detection.score_weights",
        "viral_detection.pre_buffer_seconds",
        "viral_detection.post_buffer_seconds",
    ]
    # Claves reales usadas en face_detector.py
    FACE_CFG_KEYS = [
        "face_detection.border_min_ratio",
        "face_detection.border_px",
        "face_detection.require_face",
        "face_detection.face_min_neighbors",
    ]

    def __init__(self, video_path: Path, config: dict):
        self.video_path = video_path
        self.config     = config
        cache_root      = Path(config["paths"].get("cache_dir", "videos/cache"))
        self.cache_dir  = cache_root / video_path.stem
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.video_fp   = _video_fingerprint(video_path)
        self._meta      = self._load_meta()

    def _meta_path(self) -> Path:
        return self.cache_dir / "meta.json"

    def _load_meta(self) -> dict:
        p = self._meta_path()
        if p.exists():
            try:
                return json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                pass
        return {}

    def _save_meta(self) -> None:
        self._meta_path().write_text(
            json.dumps(self._meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def _is_video_valid(self) -> bool:
        return self._meta.get("video_fp") == self.video_fp

    def get_transcription(self) -> Optional[List[Dict]]:
        if not self._is_video_valid():
            logger.debug("Caché invalidada: el vídeo ha cambiado.")
            return None
        if self._meta.get("transcription_cfg_fp") != _config_fingerprint(self.config, self.TRANSCRIPTION_CFG_KEYS):
            logger.debug("Caché invalidada: config de Whisper cambió.")
            return None
        cache_file = self.cache_dir / "transcription.json"
        if not cache_file.exists():
            return None
        try:
            data = json.loads(cache_file.read_text(encoding="utf-8"))
            logger.info(f"Transcripción cargada desde caché ({len(data)} palabras)")
            return data
        except Exception:
            return None

    def save_transcription(self, words: List[Dict]) -> None:
        (self.cache_dir / "transcription.json").write_text(
            json.dumps(words, ensure_ascii=False), encoding="utf-8"
        )
        self._meta["video_fp"]             = self.video_fp
        self._meta["transcription_cfg_fp"] = _config_fingerprint(self.config, self.TRANSCRIPTION_CFG_KEYS)
        self._save_meta()

    def get_segments(self) -> Optional[List[Dict]]:
        if not self._is_video_valid():
            return None
        if self._meta.get("segments_cfg_fp") != _config_fingerprint(self.config, self.SEGMENTS_CFG_KEYS):
            logger.debug("Caché invalidada: config de scoring cambió.")
            return None
        cache_file = self.cache_dir / "segments.json"
        if not cache_file.exists():
            return None
        try:
            data = json.loads(cache_file.read_text(encoding="utf-8"))
            logger.info(f"Segmentos cargados desde caché ({len(data)} segmentos)")
            return data
        except Exception:
            return None

    def save_segments(self, segments: List[Dict]) -> None:
        (self.cache_dir / "segments.json").write_text(
            json.dumps(segments, ensure_ascii=False), encoding="utf-8"
        )
        self._meta["segments_cfg_fp"] = _config_fingerprint(self.config, self.SEGMENTS_CFG_KEYS)
        self._save_meta()

    def get_face_data(self, clip_stem: str) -> Optional[List[Dict]]:
        if not self._is_video_valid():
            return None
        cfg_fp = _config_fingerprint(self.config, self.FACE_CFG_KEYS)
        if self._meta.get(f"face_cfg_fp_{clip_stem}") != cfg_fp:
            return None
        cache_file = self.cache_dir / f"face_{clip_stem}.json"
        if not cache_file.exists():
            return None
        try:
            data = json.loads(cache_file.read_text(encoding="utf-8"))
            logger.info(f"Datos faciales cargados desde caché: {clip_stem}")
            return data
        except Exception:
            return None

    def save_face_data(self, clip_stem: str, face_data: List[Dict]) -> None:
        (self.cache_dir / f"face_{clip_stem}.json").write_text(
            json.dumps(face_data, ensure_ascii=False), encoding="utf-8"
        )
        self._meta[f"face_cfg_fp_{clip_stem}"] = _config_fingerprint(self.config, self.FACE_CFG_KEYS)
        self._save_meta()

    def is_clip_done(self, final_path: Path) -> bool:
        return final_path.exists() and final_path.stat().st_size > 0

    def clear(self) -> None:
        import shutil
        shutil.rmtree(self.cache_dir, ignore_errors=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._meta = {}
        logger.info(f"Caché borrada: {self.cache_dir}")
