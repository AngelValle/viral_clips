# 🎬 Viral Clip Automation — Modo Local

Pipeline Python para extraer automáticamente clips virales de vídeos locales,
con composición 9:16 dinámica, subtítulos estilo TikTok y censura automática.

---

## Estructura del proyecto

```
viral_clips/
├── main.py                 # Procesa un fichero o carpeta
├── watch.py                # Vigila input/ y procesa automáticamente
├── config.json             # Configuración global
├── requirements.txt
├── modules/
│   ├── scanner.py          # Escaneo y cola de vídeos locales
│   ├── transcriber.py      # Whisper (timestamps por palabra)
│   ├── viral_scorer.py     # Scoring de momentos virales
│   ├── face_detector.py    # Detección facial con MediaPipe
│   ├── composer.py         # Composición 9:16 con ffmpeg
│   ├── censor.py           # Pitido de audio + asteriscos
│   ├── subtitles.py        # Subtítulos ASS estilo TikTok
│   └── exporter.py         # Render final + metadatos JSON
├── assets/
│   └── wordlists/
│       └── custom.txt      # Palabras extra a censurar
└── videos/
    ├── input/              # ← Pon aquí tus VODs/grabaciones
    ├── output/
    │   ├── clips/          # ← Clips .mp4 finales
    │   └── metadata/       # ← JSON con info de cada clip
    └── processed/          # ← Vídeos ya procesados
```

---

## Instalación

```bash
# 1. Instalar Python 3.10+
# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Instalar FFmpeg en el sistema
# macOS:   brew install ffmpeg
# Ubuntu:  sudo apt install ffmpeg
# Windows: choco install ffmpeg
```

---

## Uso

```bash
# Procesar un único vídeo
python main.py --file videos/input/directo_2025.mp4

# Procesar toda la carpeta input/
python main.py --folder videos/input/

# Modo vigilancia automática (detecta nuevos vídeos en input/)
python watch.py

# Usar un config personalizado
python main.py --folder videos/input/ --config mi_config.json

# Procesar sin mover a 'processed' al terminar
python main.py --folder videos/input/ --no-move
```

---

## Configuración (`config.json`)

| Clave                          | Descripción                                          |
|-------------------------------|------------------------------------------------------|
| `paths.input_dir`              | Carpeta donde están tus vídeos fuente                |
| `paths.output_dir`             | Carpeta donde se guardan los clips finales           |
| `whisper.model`                | `tiny/base/small/medium/large` (+ precisión = + lento)|
| `viral_detection.top_n_clips`  | Cuántos clips extraer por vídeo                      |
| `face_detection.face_zone_ratio` | % de altura para la zona facial (default: 0.55)   |
| `censorship.mode`              | `tiktok / youtube / instagram / twitch`              |
| `censorship.custom_words`      | Lista extra de palabras a censurar                   |
| `output.fps`                   | FPS del clip final (default: 60)                     |
| `watch_mode.enabled`           | `true` para activar vigilancia desde config          |

---

## Pipeline

```
[Vídeo local .mp4/.mkv/.mov]
        ↓
  1. Transcripción (Whisper) → timestamps por palabra
        ↓
  2. Scoring viral → top N segmentos
        ↓  (por cada segmento)
  3. Extraer segmento crudo
        ↓
  4. Detección facial frame a frame (MediaPipe)
        ↓
  5. Composición 9:16 dinámica (ffmpeg)
     ├── Con rostro:    zona facial (55%) + contenido (45%)
     └── Sin rostro:    pantalla completa
        ↓
  6. Censura: pitido de audio (ffmpeg) + asteriscos en subtítulos
        ↓
  7. Subtítulos ASS estilo TikTok (embebidos)
        ↓
  8. Render final → 1080×1920 / H.264 / 60fps / .mp4
        ↓
[videos/output/clips/nombre_clip_01.mp4]
```

---

## Notas

- El modelo Whisper `medium` equilibra velocidad y precisión. Usa `large` para mayor calidad si tienes GPU.
- El scoring viral no usa métricas del chat (modo local); se basa en audio, velocidad de habla, palabras clave y cambios de escena.
- Los clips ya procesados se mueven a `videos/processed/` para no reprocesarlos.
- Edita `assets/wordlists/custom.txt` para añadir palabras extra sin tocar el código.
