"""
subtitles.py
Genera subtitulos estilo TikTok/Shorts en formato ASS:
  - Highlight de palabra activa en rojo (<font color=red>).
  - max_line_width y max_line_count configurables.
  - Palabras censuradas aparecen con asteriscos.
  - Se embeben en el video final mediante ffmpeg.
"""

import logging
import re
from pathlib import Path
from typing import List, Dict

from modules.transcriber import build_highlighted_entries

logger = logging.getLogger(__name__)


def _ass_time(s: float) -> str:
    h   = int(s // 3600)
    m   = int((s % 3600) // 60)
    sec = s % 60
    return f"{h}:{m:02d}:{sec:05.2f}"


def _convert_highlight_tags(text: str) -> str:
    """
    Reemplaza etiquetas de highlight:
      <u>palabra</u>  →  {\\c&H0000FF&}palabra{\\c&HFFFFFF&}
    El color &H0000FF& es rojo en formato ASS (BGR).
    Tambien limpia cualquier tag HTML residual.
    """
    # <u>...</u> → color rojo en ASS (formato BGR: 0000FF = rojo)
    text = re.sub(
        r"<u>(.*?)</u>",
        r"{\\c&H0000FF&}\1{\\c&HFFFFFF&}",
        text
    )
    # Limpiar cualquier tag HTML residual
    text = re.sub(r"<[^>]+>", "", text)
    return text


def write_ass(entries: List[Dict], output_path: Path, config: dict) -> Path:
    """
    Escribe un fichero .ass con estilo TikTok:
      - Fuente grande, negrita, borde negro.
      - Palabra activa resaltada en rojo.
      - Posicionado en la zona inferior del frame.
    """
    cfg_s         = config["subtitles"]
    font_size     = cfg_s["font_size"]
    outline_w     = cfg_s["outline_width"]
    pos_y_ratio   = cfg_s["position_y_ratio"]
    res_w         = config["output"]["resolution_w"]
    res_h         = config["output"]["resolution_h"]
    pos_y         = int(res_h * pos_y_ratio)
    font_name     = cfg_s.get("font_name", "Showcard Gothic")

    # Colores ASS en formato BGR hex
    font_color    = "&H00FFFFFF"  # blanco
    outline_color = "&H00000000"  # negro
    # Shadow desactivado (0), BorderStyle=1 (outline puro sin caja)
    # Bold=1, Italic=0, Shadow=0

    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {res_w}
PlayResY: {res_h}
WrapStyle: 1

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, OutlineColour, BackColour, Bold, Italic, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV
Style: TikTok,{font_name},{font_size},{font_color},{outline_color},&H00000000,1,0,1,{outline_w},0,2,60,60,60

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

    event_lines = []
    for e in entries:
        # Convertir etiquetas <u> a color rojo ASS
        text = _convert_highlight_tags(e["text"])
        event_lines.append(
            f"Dialogue: 0,{_ass_time(e['start'])},{_ass_time(e['end'])},"
            f"TikTok,,0,0,0,,"
            f"{{\\an2\\pos({res_w // 2},{pos_y})}}{text}"
        )

    output_path.write_text(
        header + "\n".join(event_lines) + "\n",
        encoding="utf-8"
    )
    logger.info(f"ASS generado: {output_path.name} ({len(entries)} entradas)")
    return output_path


def generate_subtitles(
    words: List[Dict],
    config: dict,
    output_dir: Path,
    clip_name: str,
) -> Path:
    """
    Punto de entrada: genera el fichero .ass con highlight de palabra activa.
    Usa build_highlighted_entries para implementar:
      --highlight_words true
      --max_line_width 20
      --max_line_count 1
    """
    cfg_s          = config["subtitles"]
    max_line_width = cfg_s.get("max_line_width", 20)
    max_line_count = cfg_s.get("max_line_count", 1)

    entries  = build_highlighted_entries(
        words,
        max_line_width=max_line_width,
        max_line_count=max_line_count,
    )
    ass_path = output_dir / f"{clip_name}.ass"
    return write_ass(entries, ass_path, config)
