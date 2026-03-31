"""
log_setup.py
Formateador de logs visual para el pipeline de clips virales.

Diseño:
  - Colores ANSI compatibles con Windows 10+ (consola moderna / Windows Terminal)
  - Iconos Unicode por nivel: ✦ info · ⚠ warning · ✖ error · ◌ debug
  - Nombre de módulo abreviado y en color apagado para no distraer
  - Timestamps compactos solo con hora:min:seg
  - Mensajes especiales reconocidos por prefijo para formateo extra
    (cabeceras de paso, separadores, resultados)
"""

import logging
import re
import sys


# ── Paleta ANSI ───────────────────────────────────────────────────────────────

class C:
    RESET   = "\033[0m"
    BOLD    = "\033[1m"
    DIM     = "\033[2m"

    # Colores de texto
    WHITE   = "\033[97m"
    GREY    = "\033[90m"
    CYAN    = "\033[96m"
    GREEN   = "\033[92m"
    YELLOW  = "\033[93m"
    RED     = "\033[91m"
    MAGENTA = "\033[95m"
    BLUE    = "\033[94m"
    ORANGE  = "\033[38;5;208m"

    # Fondos (para cabeceras)
    BG_DARK = "\033[48;5;235m"


# ── Abreviaciones de módulos ──────────────────────────────────────────────────

_MODULE_ABBR = {
    "main":               "main",
    "modules.transcriber":  "whisper",
    "modules.viral_scorer": "scorer",
    "modules.face_detector":"faces",
    "modules.composer":     "composer",
    "modules.exporter":     "exporter",
    "modules.censor":       "censor",
    "modules.subtitles":    "subs",
    "modules.neon_name":    "neon",
    "modules.cache":        "cache",
    "modules.scanner":      "scanner",
    "modules.gpu_utils":    "gpu",
    "modules.metadata_generator": "metadata",
}

def _abbr(name: str) -> str:
    return _MODULE_ABBR.get(name, name.split(".")[-1])


# ── Formatter ─────────────────────────────────────────────────────────────────

class PipelineFormatter(logging.Formatter):
    """
    Formatter con colores, iconos y reconocimiento de mensajes especiales.

    Mensajes especiales (por prefijo en el texto):
      "PASO N/M — ..."   → cabecera de paso con número y título resaltados
      "SEPARADOR"        → línea horizontal
      "✔ ..."            → resultado final verde
    """

    LEVEL_STYLE = {
        logging.DEBUG:    (C.GREY,    "◌"),
        logging.INFO:     (C.CYAN,    "✦"),
        logging.WARNING:  (C.YELLOW,  "⚠"),
        logging.ERROR:    (C.RED,     "✖"),
        logging.CRITICAL: (C.RED + C.BOLD, "✖"),
    }

    # Patrones especiales en el texto del mensaje
    _RE_PASO   = re.compile(r"^PASO\s+(\d+)/(\d+)\s+[—–-]\s+(.+)$")
    _RE_CLIP   = re.compile(r"^Clip\s+(\d+)/(\d+)\s+\[(.+?)\]\s+score=([\d.]+)$")
    _RE_OK     = re.compile(r"^[✔✓☑]\s+(.+)$")
    _RE_CACHED = re.compile(r"^(PASO\s+\d+/\d+\s+[—–-]\s+.+?)\s+✓\s+\(desde caché\)$")
    _RE_SEP    = re.compile(r"^[═─=\-]{10,}$")

    def format(self, record: logging.LogRecord) -> str:
        color, icon = self.LEVEL_STYLE.get(record.levelno, (C.WHITE, "·"))
        ts          = self.formatTime(record, "%H:%M:%S")
        module      = _abbr(record.name)
        msg         = record.getMessage()

        # ── Separador ────────────────────────────────────────────────────────
        if self._RE_SEP.match(msg):
            return f"{C.DIM}{C.GREY}{'─' * 60}{C.RESET}"

        # ── Cabecera de vídeo (═══ Procesando ═══) ───────────────────────────
        if msg.startswith("Procesando:"):
            name = msg.replace("Procesando:", "").strip()
            bar  = "═" * 58
            return (
                f"\n{C.BOLD}{C.MAGENTA}{bar}{C.RESET}\n"
                f"  {C.BOLD}{C.WHITE}▶  {name}{C.RESET}\n"
                f"{C.BOLD}{C.MAGENTA}{bar}{C.RESET}"
            )

        # ── Paso del pipeline ─────────────────────────────────────────────────
        m = self._RE_PASO.match(msg)
        if m:
            n, total, title = m.group(1), m.group(2), m.group(3)
            badge = f"{C.BOLD}{C.BG_DARK}{C.CYAN} {n}/{total} {C.RESET}"
            return (
                f"\n{C.DIM}{C.GREY}{ts}{C.RESET}  "
                f"{badge}  {C.BOLD}{C.WHITE}{title}{C.RESET}"
            )

        # ── Paso desde caché ──────────────────────────────────────────────────
        m = self._RE_CACHED.match(msg)
        if m:
            inner = self._RE_PASO.match(m.group(1))
            if inner:
                n, total, title = inner.group(1), inner.group(2), inner.group(3)
                badge  = f"{C.BOLD}{C.BG_DARK}{C.GREY} {n}/{total} {C.RESET}"
                cached = f"{C.DIM}{C.GREEN}  ↩ caché{C.RESET}"
                return (
                    f"\n{C.DIM}{C.GREY}{ts}{C.RESET}  "
                    f"{badge}  {C.DIM}{C.WHITE}{title}{cached}{C.RESET}"
                )

        # ── Separador de clip ─────────────────────────────────────────────────
        m = self._RE_CLIP.match(msg)
        if m:
            n, total, rng, score = m.group(1), m.group(2), m.group(3), m.group(4)
            return (
                f"\n{C.DIM}{C.GREY}{'·' * 60}{C.RESET}\n"
                f"  {C.BOLD}{C.ORANGE}Clip {n}{C.RESET}{C.DIM}{C.GREY}/{total}{C.RESET}"
                f"  {C.YELLOW}[{rng}]{C.RESET}"
                f"  {C.DIM}score {C.CYAN}{score}{C.RESET}"
            )

        # ── Resultado positivo (✔ / ✓) ───────────────────────────────────────
        m = self._RE_OK.match(msg)
        if m:
            return (
                f"{C.DIM}{C.GREY}{ts}{C.RESET}  "
                f"{C.BOLD}{C.GREEN}✔{C.RESET}  {C.GREEN}{m.group(1)}{C.RESET}"
            )

        # ── WARNING ───────────────────────────────────────────────────────────
        if record.levelno == logging.WARNING:
            return (
                f"{C.DIM}{C.GREY}{ts}{C.RESET}  "
                f"{C.YELLOW}⚠{C.RESET}  {C.YELLOW}{msg}{C.RESET}"
                f"  {C.DIM}{C.GREY}[{module}]{C.RESET}"
            )

        # ── ERROR ─────────────────────────────────────────────────────────────
        if record.levelno >= logging.ERROR:
            return (
                f"{C.DIM}{C.GREY}{ts}{C.RESET}  "
                f"{C.BOLD}{C.RED}✖{C.RESET}  {C.BOLD}{C.RED}{msg}{C.RESET}"
                f"  {C.DIM}{C.GREY}[{module}]{C.RESET}"
            )

        # ── INFO estándar ─────────────────────────────────────────────────────
        # Detectar sub-mensajes de progreso (empiezan con espacios o [N/M])
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


# ── Setup público ─────────────────────────────────────────────────────────────

def setup_logging(level: int = logging.INFO) -> None:
    """
    Configura el logging del pipeline con el formatter visual.
    Llama a esto una sola vez al inicio de main.py.
    """
    # Activar colores ANSI en Windows (requiere Python 3.12+ o Windows 10 1909+)
    if sys.platform == "win32":
        try:
            import ctypes
            kernel32 = ctypes.windll.kernel32
            # ENABLE_VIRTUAL_TERMINAL_PROCESSING = 0x0004
            kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
        except Exception:
            pass

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(PipelineFormatter())

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)

    # Silenciar librerías ruidosas
    for noisy in ("faster_whisper", "torch", "PIL", "urllib3",
                   "httpx", "httpcore", "anthropic"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
