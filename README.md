# Videos2ViralShort — Obtener clips/short desde videos/directos

Pipeline automatizado para extraer, editar y publicar clips virales de streams/videos en formato vertical 9:16 (TikTok/YouTube Shorts) y horizontal 16:9. Transcripción con Whisper, detección viral con Gemini AI o señales de audio, composición GPU con FFmpeg NVENC y subtítulos estilo TikTok con karaoke.

---

## Tabla de Contenidos

- [Características](#características)
- [Requisitos del sistema](#requisitos-del-sistema)
- [Instalación](#instalación)
- [Variables de entorno](#variables-de-entorno)
- [¿Cómo configurar el proyecto?](#cómo-configurar-el-proyecto)
- [Estructura del proyecto](#estructura-del-proyecto)
- [Configuración](#configuración)
- [Uso](#uso)
- [Pipeline — Pasos detallados](#pipeline--pasos-detallados)
- [Módulos](#módulos)
- [Interfaz web (UI)](#interfaz-web-ui)
- [Herramientas de calibración](#herramientas-de-calibración)
- [Publicación automática](#publicación-automática)
- [Caché del pipeline](#caché-del-pipeline)

---

## Características

- **Transcripción automática** con `faster-whisper` (modelo `large-v3`, GPU float16)
- **Diarización de locutores** opcional con Pyannote Audio 3.1 (requiere `HF_TOKEN`)
- **Detección viral** con Google Gemini (analiza la transcripción completa del stream) o fallback por señales de audio (picos de energía, velocidad de habla, palabras clave)
- **Composición 9:16** dinámica: gameplay centrado + overlay de webcam con animación neon del nombre del streamer (modo Vertical). Modo **Horizontal 16:9** disponible para conservar el vídeo original sin reencuadre
- **Detección de modo conducción** automática (ajusta el recorte del HUD de GTA)
- **Subtítulos estilo TikTok** con efecto karaoke y colores por locutor (formato ASS)
- **Censura automática** de palabras malsonantes con pitido (perfiles: tiktok, youtube, instagram, twitch, o desactivado)
- **Caché inteligente** por vídeo: evita reprocesar transcripciones, segmentos y datos faciales si no hay cambios
- **Modo watcher** (`--watch`): monitoriza la carpeta de entrada y procesa nuevos vídeos automáticamente
- **Publicación automática** a YouTube Shorts y TikTok con soporte de programación horaria
- **Interfaz web Streamlit** para revisar cortes, ajustar tiempos, lanzar el pipeline y configurar todos los parámetros sin editar archivos
- **GPU end-to-end**: NVDEC para decodificación, NVENC (h264_nvenc) para codificación

---

## Requisitos del sistema

| Componente | Recomendado |
|---|---|
| SO | Windows 10/11 |
| GPU | NVIDIA RTX (con soporte NVENC/NVDEC) |
| VRAM | 8 GB mín. — 16 GB para Whisper large-v3 |
| RAM | 16 GB |
| Python | 3.11 – 3.13 |
| CUDA | 12.8+ |
| FFmpeg | Build completo con soporte NVENC |

---

## Instalación

### 1. Clonar el repositorio

```bash
git clone <repo-url>
cd viral_clips_v1
```

### 2. Instalar dependencias base

```bash
pip install -r requirements.txt
```

### 3. Instalar PyTorch con CUDA 12.8

> **Obligatorio para GPU.** No usar `pip install torch` sin el índice CUDA (instala versión CPU-only).

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

### 4. Instalar FFmpeg con NVENC

Descargar un build completo desde https://ffmpeg.org/download.html y añadir al PATH del sistema.

Verificar que NVENC está disponible:
```bash
ffmpeg -encoders | findstr nvenc
```

### 5. Playwright (solo para auto-publicación TikTok)

```bash
playwright install
```

---

## Variables de entorno

Las claves API se leen **exclusivamente** desde variables de entorno. Nunca se almacenan en `config.json` ni en ningún archivo del repositorio.

Añadir al `$PROFILE` de PowerShell (`notepad $PROFILE`):

```powershell
$env:GEMINI_API_KEY                  = "tu_clave_gemini"
$env:HF_TOKEN                        = "tu_token_huggingface"
$env:PYTHONIOENCODING                = "utf-8"
$env:PYTHONUTF8                      = "1"
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
```

| Variable | Descripción | Obligatoria |
|---|---|---|
| `GEMINI_API_KEY` | Clave de Google AI Studio para detección viral con Gemini y generación de metadata | No (el pipeline usa fallback por señales) |
| `HF_TOKEN` | Token de Hugging Face para diarización Pyannote | No (usa locutor único si no está definido) |

Obtener claves:
- **Gemini**: https://aistudio.google.com/apikey
- **Hugging Face**: https://huggingface.co/settings/tokens — aceptar además los términos del modelo en https://hf.co/pyannote/speaker-diarization-3.1

---

## ¿Cómo configurar el proyecto?

Sigue estos pasos en orden la primera vez que configures el proyecto en una máquina nueva.

---

### Paso 1 — Completar la instalación

Asegúrate de haber completado los pasos de la sección [Instalación](#instalación):

```bash
pip install -r requirements.txt
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

Y de haber añadido FFmpeg (con NVENC) al PATH del sistema.

---

### Paso 2 — Configurar las variables de entorno

Añade las claves al `$PROFILE` de PowerShell (`notepad $PROFILE`):

```powershell
$env:GEMINI_API_KEY = "tu_clave_de_google_ai_studio"
$env:HF_TOKEN       = "tu_token_de_huggingface"        # Opcional
```

Reinicia el terminal para que surtan efecto. Verificar que están activas:

```powershell
echo $env:GEMINI_API_KEY
```

---

### Paso 3 — Verificar GPU y FFmpeg

Ejecuta el diagnóstico para confirmar que todo el hardware funciona:

```bash
python tools/calibrate.py gpu
```

Resultado esperado:
- `nvidia-smi` muestra la GPU
- `CUDA disponible: True` en PyTorch
- Encoders NVENC listados (al menos `h264_nvenc`)
- `✔  h264_nvenc funciona correctamente`

Si `h264_nvenc` falla, revisa que tu build de FFmpeg incluye encoders NVENC (los builds oficiales de ffmpeg.org los incluyen).

---

### Paso 4 — Calibrar la zona de la webcam

Necesitas un vídeo del stream donde la webcam sea visible. Elige un segundo en el que la cara esté bien encuadrada:

```bash
python tools/calibrate.py webcam --file videos/input/tu_stream.mp4 --second 120
```

Se abrirá una ventana OpenCV. **Haz clic y arrastra** sobre la zona de la webcam y pulsa **ENTER**. El script imprimirá valores como:

```
"webcam_w_ratio":        0.137,
"webcam_h_ratio":        0.331,
"webcam_x_offset":       22,
"webcam_y_center_ratio": 0.500,
```

Copia esos cuatro valores en `config.json → layout`.

---

### Paso 5 — Calibrar el border ratio

El pipeline detecta si la webcam está visible analizando el color del borde (morado por defecto). Necesitas medir el ratio correcto para tu stream:

```bash
# Segundo donde la cámara SÍ está visible
python tools/calibrate.py border --file tu_stream.mp4 --second 120

# Opcional: comparar con un segundo donde NO está la cámara
python tools/calibrate.py border --file tu_stream.mp4 --second 120 --second2 600
```

El script mostrará el `border_min_ratio` medido y recomendaciones. Copia el valor sugerido en `config.json → face_detection → border_min_ratio`.

> Si el borde de tu webcam tiene un color diferente al morado, ajusta también `border_color_hsv_lower` y `border_color_hsv_upper` en `face_detection`.

---

### Paso 6 — Ajustar los parámetros del streamer

La forma más cómoda es usar la **Pestaña 5 — Configuración** de la UI, que permite editar todos los parámetros sin tocar archivos JSON. Si prefieres editar directamente `config.json`:

```json
"claude": {
  "streamer_name": "TuNombreAqui",
  "game_name":     "GTA V"
},
"whisper": {
  "language": "es"
},
"viral_detection": {
  "skip_intro_sec": 30,
  "skip_outro_sec": 30
}
```

- `streamer_name`: aparece en la animación neon del overlay
- `language`: código ISO del idioma del stream (`"es"`, `"en"`, etc.)
- `skip_intro_sec/outro_sec`: ajusta según la duración real de tu intro/outro

---

### Paso 7 — Primer test

Ejecuta solo hasta el Paso 2 para ver los segmentos detectados sin renderizar nada:

```bash
python main.py --file videos/input/tu_stream.mp4 --max-step 2
```

Luego abre la UI para revisar los cortes:

```bash
streamlit run ui.py
# o doble clic en iniciar.bat
```

Ve a la **Pestaña 2 — Revisión y Publicación**, selecciona el vídeo y comprueba que los segmentos tienen sentido. Ajusta los tiempos si es necesario y guarda.

---

### Paso 8 — Render completo del primer clip

Si los segmentos son correctos, lanza el pipeline completo:

```bash
python main.py --file videos/input/tu_stream.mp4
```

O desde la UI (Pestaña 1), selecciona el vídeo con paso máximo **8 — Pipeline Completo** y pulsa **Lanzar**.

Los clips finales aparecerán en `videos/output/clips/`.

---

### Paso 9 — (Opcional) Configurar publicación automática

**YouTube Shorts:**
1. Crea un proyecto en [Google Cloud Console](https://console.cloud.google.com)
2. Habilita **YouTube Data API v3**
3. Crea credenciales OAuth2 (tipo Desktop) → descarga `client_secrets.json` en la raíz del proyecto
4. La primera vez que el Paso 8 intente publicar en YouTube, abrirá el navegador para autorización

**TikTok:**
1. Instala Playwright: `playwright install`
2. Exporta tus cookies de TikTok a `cookies.txt` (formato Netscape) usando una extensión de navegador
3. Coloca `cookies.txt` en la raíz del proyecto

---

### Resumen rápido de configuración

| # | Acción | Herramienta |
|---|---|---|
| 1 | Instalar dependencias + PyTorch CUDA | `pip install` |
| 2 | Añadir `GEMINI_API_KEY` y `HF_TOKEN` al perfil | PowerShell `$PROFILE` |
| 3 | Verificar GPU y NVENC | `calibrate.py gpu` |
| 4 | Calibrar zona webcam | `calibrate.py webcam` |
| 5 | Calibrar border ratio | `calibrate.py border` |
| 6 | Ajustar nombre streamer, idioma, intro/outro | `config.json` |
| 7 | Test con `--max-step 2` + revisión en UI | `main.py` + `ui.py` |
| 8 | Render completo | `main.py` o UI |
| 9 | (Opcional) Auth YouTube / cookies TikTok | `client_secrets.json` / `cookies.txt` |

---

## Estructura del proyecto

```
viral_clips_v1/
│
├── main.py                  # Punto de entrada CLI
├── ui.py                    # Interfaz web Streamlit
├── config.json              # Configuración global (sin claves API)
├── requirements.txt         # Dependencias Python
├── iniciar.bat              # Lanzador de la UI en Windows
│
├── modules/
│   ├── scanner.py           # Escaneo de carpeta de entrada y cola de vídeos
│   ├── transcriber.py       # Transcripción Whisper + diarización Pyannote + generación ASS
│   ├── viral_scorer.py      # Detección de momentos virales (Gemini + fallback señales)
│   ├── face_detector.py     # Detección facial OpenCV + detección modo conducción
│   ├── composer.py          # Composición 9:16 con FFmpeg (gameplay + webcam + neon)
│   ├── exporter.py          # Extracción de segmentos, embed subtítulos, metadata Gemini
│   ├── censor.py            # Detección y censura de palabras malsonantes
│   ├── cache.py             # Caché por vídeo con invalidación por fingerprint de config
│   ├── gpu_utils.py         # Detección de hardware GPU y configuración de FFmpeg
│   └── publisher.py         # Publicación a YouTube Shorts y TikTok
│
├── tools/
│   ├── calibrate.py         # Herramientas de calibración (gpu, webcam, border)
│   ├── find_webcam_coords.py
│   └── measure_border_ratio.py
│
├── assets/
│   └── haarcascade_frontalface_default.xml   # Modelo Haar para detección facial
│
└── videos/
    ├── input/               # Vídeos de entrada (.mp4, .mkv, .mov)
    ├── output/
    │   ├── clips/           # Clips finales generados
    │   └── metadata/        # JSON con títulos, descripción y hashtags por clip
    ├── processed/           # Vídeos ya procesados (movidos automáticamente)
    └── cache/               # Caché del pipeline (transcripciones, segmentos, caras)
```

---

## Configuración

El archivo `config.json` controla todos los parámetros del pipeline.

### `paths`

```json
{
  "input_dir": "videos/input/",
  "output_dir": "videos/output/",
  "processed_dir": "videos/processed/",
  "cache_dir": "videos/cache",
  "supported_formats": [".mp4", ".mkv", ".mov"]
}
```

### `whisper`

```json
{
  "model": "large-v3",
  "language": "es",
  "device": "auto",
  "compute_type": "float16"
}
```

| Parámetro | Descripción |
|---|---|
| `model` | Modelo Whisper. Opciones: `tiny`, `base`, `small`, `medium`, `large-v3` |
| `language` | Código ISO del idioma (ej. `"es"`, `"en"`) |
| `device` | `"auto"` detecta CUDA. Forzar con `"cuda"` o `"cpu"` |
| `compute_type` | `"float16"` para GPU, `"int8"` para CPU |

### `viral_detection`

```json
{
  "min_clip_duration": 10,
  "max_clip_duration": 60,
  "top_n_clips": 999,
  "pre_buffer_seconds": 3,
  "post_buffer_seconds": 2,
  "skip_intro_sec": 28,
  "skip_outro_sec": 28,
  "viral_keywords": ["increíble", "vamos", "brutal", ...]
}
```

| Parámetro | Descripción |
|---|---|
| `min/max_clip_duration` | Duración mínima/máxima de cada clip en segundos |
| `top_n_clips` | Máximo de clips a extraer por stream (`999` = sin límite) |
| `pre/post_buffer_seconds` | Segundos de margen antes/después de cada momento viral |
| `skip_intro/outro_sec` | Segundos a ignorar al principio y final del stream |
| `viral_keywords` | Palabras que suman puntuación en el fallback por señales |

### `face_detection`

```json
{
  "enabled": true,
  "border_min_ratio": 0.7,
  "border_px": 8,
  "border_color_hsv_lower": [125, 50, 50],
  "border_color_hsv_upper": [165, 255, 255],
  "require_face": true,
  "face_min_neighbors": 9,
  "face_min_size_ratio": 0.2
}
```

| Parámetro | Descripción |
|---|---|
| `enabled` | Si `false`, omite completamente el análisis OpenCV y renderiza en modo pantalla completa sin overlay de webcam |
| `border_min_ratio` | Fracción mínima de píxeles del color del borde para confirmar webcam visible. **Calibrar con `tools/calibrate.py border`** |
| `border_color_hsv_lower/upper` | Rango HSV del color del borde de la webcam |
| `border_px` | Grosor en píxeles del borde a analizar |
| `require_face` | Si `true`, solo activa overlay webcam cuando detecta cara |

### `layout`

```json
{
  "detect_driving": true,
  "webcam_w_ratio": 0.137,
  "webcam_h_ratio": 0.331,
  "webcam_x_offset": 22,
  "webcam_y_center_ratio": 0.5,
  "gameplay_zoom": 1.5,
  "webcam_overlay_w_ratio": 0.485,
  "webcam_overlay_y_ratio": 0.01,
  "gameplay_x_offset": -200,
  "gameplay_x_offset_driving": -150
}
```

**Calibrar con `tools/calibrate.py webcam`.**

| Parámetro | Descripción |
|---|---|
| `detect_driving` | Si `false`, deshabilita la detección de modo conducción y usa siempre el layout estándar |
| `webcam_w/h_ratio` | Anchura/altura de la zona webcam como fracción del frame |
| `webcam_x_offset` | Offset horizontal en píxeles del borde izquierdo de la webcam |
| `webcam_y_center_ratio` | Centro vertical de la webcam como fracción de la altura |
| `gameplay_zoom` | Factor de zoom del gameplay en el layout 9:16 |
| `gameplay_x_offset` | Desplazamiento horizontal del gameplay (centrar juego) |
| `gameplay_x_offset_driving` | Offset en modo conducción (HUD de vehículo) |

### `subtitles`

```json
{
  "enabled": true,
  "font_size": 72,
  "font_name": "Showcard Gothic",
  "outline_width": 4,
  "position_y_ratio": 0.82,
  "max_line_width": 20,
  "max_line_count": 1,
  "manual_offset_sec": 0.4
}
```

| Parámetro | Descripción |
|---|---|
| `enabled` | Si `false`, omite la generación y el embed del `.ass` — el clip se guarda sin subtítulos |
| `font_name` | Nombre de la fuente instalada en el sistema |
| `position_y_ratio` | Posición vertical (0.0 = arriba, 1.0 = abajo) |
| `max_line_width` | Número máximo de caracteres por frase en el karaoke |
| `manual_offset_sec` | Ajuste fino de sincronía en segundos (positivo = retrasa subtítulos) |

### `censorship`

```json
{
  "mode": "tiktok",
  "custom_words": [],
  "beep_frequency_hz": 1000,
  "profiles": {
    "tiktok":     {"strictness": "high"},
    "youtube":    {"strictness": "medium"},
    "instagram":  {"strictness": "high"},
    "twitch":     {"strictness": "low"}
  }
}
```

| Parámetro | Descripción |
|---|---|
| `mode` | Perfil de censura activo: `"tiktok"`, `"youtube"`, `"instagram"`, `"twitch"`, `"desactivado"`. Con `"desactivado"` no se censura ninguna palabra ni se aplica pitido |
| `custom_words` | Lista adicional de palabras a censurar |
| `beep_frequency_hz` | Frecuencia del pitido de censura en Hz |

### `transcriber`

```json
{
  "diarization_enabled": true
}
```

| Parámetro | Descripción |
|---|---|
| `diarization_enabled` | Si `false`, deshabilita Pyannote aunque `HF_TOKEN` esté definido. Todos los subtítulos usan el color por defecto (blanco) |

### `gemini`

```json
{
  "model": "gemini-3.1-flash-lite-preview"
}
```

Cambiar el modelo según disponibilidad en Google AI Studio. La clave API siempre va en la variable de entorno `GEMINI_API_KEY`.

### `ai_features`

```json
{
  "multimodal_video": false,
  "multimodal_interval_sec": 10
}
```

| Parámetro | Descripción |
|---|---|
| `multimodal_video` | Si `true`, extrae frames JPEG del stream y los envía inline junto a la transcripción en la llamada a Gemini del Paso 2. Permite detectar momentos virales también por contexto visual. Compatible con el tier gratuito (sin Files API). Incluye reintentos automáticos ante rate limit (5 intentos × 10 s) |
| `multimodal_interval_sec` | Cada cuántos segundos del stream se extrae un frame. Valor por defecto: `10`. Valores más bajos = más contexto visual pero más tokens. Rango recomendado: 5–60 s |

### `gpu`

```json
{
  "force_cpu": false,
  "ffmpeg_encoder_override": null,
  "torch_device_override": null
}
```

| Parámetro | Descripción |
|---|---|
| `force_cpu` | Forzar procesamiento en CPU (deshabilita CUDA) |
| `ffmpeg_encoder_override` | Forzar encoder FFmpeg concreto (ej. `"libx264"` para CPU) |
| `torch_device_override` | Forzar dispositivo torch (ej. `"cpu"`) |

### `output`

```json
{
  "resolution_w": 1080,
  "resolution_h": 1920,
  "fps": 60,
  "audio_codec": "aac",
  "orientation": "vertical",
  "naming_pattern": "{source_name}_clip_{n}.mp4"
}
```

| Parámetro | Descripción |
|---|---|
| `resolution_w/h` | Resolución de salida. Por defecto 1080×1920 (vertical 9:16) |
| `fps` | FPS del vídeo de salida |
| `orientation` | `"vertical"` — aplica composición 9:16 con webcam overlay (Paso 5). `"horizontal"` — omite la composición y conserva el vídeo en su resolución original; la resolución del ASS de subtítulos se auto-detecta con ffprobe |
| `naming_pattern` | Patrón de nombre de los clips finales. Variables disponibles: `{source_name}`, `{n}` |

---

## Uso

### Procesar un único vídeo

```bash
python main.py --file videos/input/stream.mp4
```

### Procesar una carpeta completa

```bash
python main.py --folder videos/input/
```

### Modo vigilancia automática

Monitoriza `input_dir` y procesa automáticamente cualquier vídeo nuevo:

```bash
python main.py --watch
```

### Ejecutar solo hasta un paso concreto

Útil para revisar los segmentos detectados antes de renderizar:

```bash
# Solo transcribir y detectar momentos virales
python main.py --file stream.mp4 --max-step 2

# Revisar en la UI (Pestaña 2) y luego renderizar
python main.py --file stream.mp4 --max-step 8
```

### Referencia completa de argumentos

| Argumento | Descripción |
|---|---|
| `--file <ruta>` | Procesar un único vídeo |
| `--folder <ruta>` | Procesar todos los vídeos de una carpeta |
| `--watch` | Modo vigilancia automática de `input_dir` |
| `--config <ruta>` | Ruta alternativa a `config.json` (por defecto: junto al script) |
| `--max-step <1-8>` | Ejecutar solo hasta el paso indicado (ver tabla de pasos) |
| `--no-cache` | Ignorar caché y reprocesar todo desde cero |
| `--clear-cache` | Borrar caché del vídeo antes de procesar |
| `--no-move` | No mover vídeos a `processed/` al terminar |

### Interfaz web

```bash
# Windows
iniciar.bat

# Manual
streamlit run ui.py
```

---

## Pipeline — Pasos detallados

El pipeline se divide en 8 pasos controlables con `--max-step`. Cada paso usa la caché del anterior para no repetir trabajo.

### Paso 1 — Transcripción con Whisper

- Modelo configurable (por defecto `large-v3` en CUDA float16)
- Parámetros optimizados para precisión temporal: `best_of=5`, `beam_size=5`, `condition_on_previous_text=False`, `chunk_length=30`, `max_new_tokens=128`
- Cada palabra incluye: texto, `start`, `end`, `confidence`, `speaker`
- **Diarización opcional** (requiere `HF_TOKEN`): Pyannote Audio 3.1 asigna un `speaker_id` a cada palabra. El audio se extrae a mono 16 kHz con FFmpeg y se procesa en GPU.
- Resultado cacheado en `transcription.json`

### Paso 2 — Detección de momentos virales

**Con `GEMINI_API_KEY`:**
- Envía la transcripción completa al modelo Gemini configurado
- Gemini identifica los momentos con mayor narrativa, emoción o humor
- Devuelve segmentos con rango de tiempo, puntuación (0.0–1.0) y descripción
- **Con `ai_features.multimodal_video = true`**: además de la transcripción, extrae frames JPEG (480px) del stream cada `multimodal_interval_sec` segundos (por defecto 10 s) y los envía **inline** junto al prompt (sin Files API — compatible con el tier gratuito). Gemini puede así detectar momentos virales también por contexto visual (gameplay, reacciones en pantalla, etc.). Si algún frame falla se omite sin interrumpir. En caso de rate limit del tier gratuito, reintenta automáticamente hasta 5 veces con 10 s de espera. El intervalo es configurable desde la UI (Pestaña 5 — Configuración).

**Sin `GEMINI_API_KEY` (fallback automático):**
- Analiza el audio del vídeo: picos de energía RMS y velocidad de habla (palabras/segundo)
- Detecta menciones de `viral_keywords` en la transcripción
- Pondera señales: `audio_peak × 0.45 + speech_speed × 0.35 + keywords × 0.20`

Los segmentos pueden ser **multi-fragmento**: varios rangos de tiempo discontinuos que se concatenan en un solo clip. Los fragmentos dentro de un mismo clip se ordenan por tiempo y no pueden solaparse entre sí (se descarta el solapante posterior).

Resultado cacheado en `segments.json`.

### Paso 3 — Extracción del segmento crudo

- FFmpeg con NVDEC (decodificación GPU) cuando está disponible
- Extrae exactamente el rango de tiempo indicado (respetando fragments si los hay)
- Recodifica en H.264 para garantizar precisión de corte en keyframe
- Archivo temporal en directorio seguro (eliminado al terminar)

### Paso 4 — Detección facial

- Si `face_detection.enabled = false`: se omite completamente este paso, el clip se compondrá en modo pantalla completa sin overlay de webcam
- **Detección de borde**: analiza la zona configurada (HSV) para confirmar si la webcam está visible en este frame
- **Detección de cara**: OpenCV Haar cascade dentro de la zona de webcam
- Analiza un frame cada 2 segundos, interpola entre muestras
- **Detección modo conducción**: analiza el HUD de GTA (componentes blancos del velocímetro/minimapa) para determinar si el jugador va en vehículo. Se puede deshabilitar con `layout.detect_driving = false`
- Resultado cacheado en `face_{clip_stem}.json` con flag `is_driving`

### Paso 5 — Composición dinámica

**Modo vertical (`output.orientation = "vertical"`, por defecto):**
- Redimensiona el gameplay a 1080×1920 (zoom configurable)
- Superpone la webcam en la posición calibrada cuando la cara está visible
- Genera animación neon del nombre del streamer (barrido de color, configurable)
- Modo conducción activo: ajusta el offset horizontal del gameplay para centrar el HUD del vehículo
- Procesado completamente por FFmpeg con h264_nvenc (GPU)

**Modo horizontal (`output.orientation = "horizontal"`):**
- Omite toda la composición; el clip extraído en bruto se usa directamente
- La resolución del vídeo de salida es la del stream original
- El `.ass` de subtítulos del Paso 6 se genera con las dimensiones reales del vídeo (auto-detectadas con ffprobe)

### Paso 6 — Censura + subtítulos + render final

1. Detecta palabras malsonantes según el perfil activo de `config.json`
2. Aplica censura de audio (pitido de 440 ms sobre cada palabra detectada)
3. Si `subtitles.enabled = true` (por defecto):
   - Genera archivo `.ass` con subtítulos estilo karaoke:
     - Frase completa visible durante toda su duración
     - Palabra activa destacada en rojo, resto en blanco (o color del locutor)
     - Colores por locutor si `transcriber.diarization_enabled = true` y `HF_TOKEN` está definido
     - Fuente `Showcard Gothic`, tamaño 72, outline negro de 4px
   - Embebe los subtítulos en el vídeo final (`ass` filter de FFmpeg)
4. Si `subtitles.enabled = false`: copia el clip censurado directamente sin procesar subtítulos
5. Guarda metadata básica en `videos/output/metadata/{clip_name}.json`

### Paso 7 — Metadata viral con IA

- Requiere `GEMINI_API_KEY`
- Envía la transcripción del clip a Gemini para generar:
  - Títulos optimizados para YouTube Shorts y TikTok
  - Descripción
  - Hashtags relevantes
- Guarda en el JSON de metadata del clip
- Sin API key: el paso se omite sin interrumpir el pipeline

### Paso 8 — Auto-publicación

- Lee el JSON de metadata para comprobar si `publish_yt` o `publish_tk` están activados (configurados en la UI)
- **YouTube Shorts**: sube vía YouTube Data API v3 con OAuth2 (`client_secrets.json`)
- **TikTok**: sube vía Playwright automatizando el navegador (`cookies.txt`)
- Soporta publicación programada: el campo `schedule_time` (ISO 8601) se pasa a la API de YouTube (`publishAt`) o al scheduler de Playwright para TikTok

---

## Módulos

| Módulo | Responsabilidad principal |
|---|---|
| `scanner.py` | Escanea `input_dir`, filtra vídeos válidos (tamaño mínimo, formato), construye la cola de procesamiento |
| `transcriber.py` | Transcripción Whisper, diarización Pyannote, generación de subtítulos ASS con karaoke y colores por locutor |
| `viral_scorer.py` | Detección de momentos virales con Gemini; fallback por análisis de señales de audio y keywords |
| `face_detector.py` | Detección de borde HSV, detección facial Haar con `lru_cache`, análisis de HUD para modo conducción |
| `composer.py` | Composición FFmpeg con filtergraph: escala gameplay, posiciona webcam, genera animación neon, aplica offsets de conducción |
| `exporter.py` | Extracción de segmentos (con NVDEC), embed de subtítulos ASS, guardado de metadata, generación de títulos con Gemini |
| `censor.py` | Carga diccionario de palabrotas por perfil de strictness, detecta ocurrencias, genera comandos FFmpeg para pitido de audio |
| `cache.py` | Caché por carpeta/vídeo con fingerprint MD5 de config para invalidación automática; almacena transcripción, segmentos y datos faciales |
| `gpu_utils.py` | Detecta GPU NVIDIA, selecciona encoder (`h264_nvenc` o `libx264`) y decoder (`cuda` o software), expone resumen de hardware |
| `publisher.py` | Publicación a YouTube con OAuth2 + `googleapiclient`; publicación a TikTok con Playwright; soporte de programación horaria |

---

## Interfaz web (UI)

Ejecutar con `iniciar.bat` o `streamlit run ui.py`. Disponible en `http://localhost:8501`.

### Pestaña 1 — Lanzador de Tareas

- Lista todos los vídeos disponibles en `input_dir`
- Para cada vídeo: checkbox de procesamiento + selector del paso máximo (1–8)
- Lanzar procesamiento en secuencia con salida en tiempo real en el Pipeline Log
- El Pipeline Log tiene auto-scroll al fondo (CSS `flex-direction: column-reverse`, sin JavaScript)

### Pestaña 2 — Revisión y Publicación

Requiere haber ejecutado al menos el **Paso 2** para que existan segmentos en caché.

- Selector de vídeo con segmentos disponibles
- Para cada clip detectado:
  - **Vista previa** generada bajo demanda con FFmpeg GPU (o CPU como fallback)
  - Score viral y descripción de Gemini
  - **Gestión de fragmentos** (fuera del formulario, con efecto inmediato):
    - Lista de fragmentos con tiempo de inicio/fin y botón **✕** para eliminar (deshabilitado si solo queda 1)
    - Botón **➕ Añadir fragmento** que añade un rango nuevo al final del último fragmento
  - **Sliders de ajuste** por fragmento: rango completo del vídeo original (0 s → duración total), calculado con ffprobe al cargar el vídeo
  - **Checkboxes de publicación**: YouTube Shorts, TikTok
  - **Programación horaria**: selector de fecha y hora para publicación diferida
- Formulario de guardado que actualiza la caché de segmentos con los ajustes
- Botón para renderizar y publicar los clips aprobados (Pasos 3–8)

### Pestaña 3 — Caché

- Vista expandible por proyecto (vídeo)
- Muestra tamaño y contenido de cada entrada de caché
- **Borrado granular**: selección individual de qué partes eliminar
  - 📝 Transcripción (`transcription.json`)
  - ✂️ Segmentos virales (`segments.json`)
  - 🎭 Datos faciales (`face_*.json` + entradas en `meta.json`)
- Borrado completo del proyecto o de toda la caché

### Pestaña 4 — Herramientas

Ver sección [Herramientas de calibración](#herramientas-de-calibración).

### Pestaña 5 — Configuración

Permite editar todos los parámetros del pipeline sin tocar `config.json` directamente. Los cambios se guardan con el botón **💾 Guardar Configuración** y surten efecto en la siguiente ejecución.

| Sección | Parámetros editables |
|---|---|
| **Streamer** | Nombre del streamer, nombre del juego, tipo de contenido |
| **Módulos opcionales** | Detección de cámara/cara, subtítulos, detección de conducción, diarización de locutores |
| **Whisper** | Modelo (`tiny` → `large-v3`), idioma, tipo de cómputo |
| **Detección viral** | Duraciones mín/máx, buffers, intro/outro, nº máximo de clips, keywords virales |
| **Subtítulos** | Fuente, tamaño, outline, posición vertical, ancho máximo de línea, offset de sincronía |
| **Censura** | Perfil activo (`tiktok`, `youtube`, `instagram`, `twitch`, `desactivado`), frecuencia del bip, palabras adicionales |
| **IA — Gemini** | Modelo Gemini, toggle multimodal + intervalo de frames configurable (5–120 s, inline, tier gratuito) |
| **GPU / Hardware** | Forzar CPU, deshabilitar CUDA |
| **Salida** | Formato de salida (`Vertical 9:16` / `Horizontal 16:9`), FPS, patrón de nombre de archivo |

---

## Herramientas de calibración

Accesibles desde la **Pestaña 4** de la UI o por línea de comandos.

### Diagnóstico GPU

```bash
python tools/calibrate.py gpu
```

Comprueba en secuencia:
1. Drivers NVIDIA (`nvidia-smi`)
2. PyTorch + CUDA (versión, disponibilidad, VRAM)
3. Encoders NVENC disponibles en FFmpeg
4. Test real de `h264_nvenc` con un vídeo de 0.2 s
5. Presencia de `nvcuda.dll` en el sistema
6. Rutas CUDA/NVIDIA en el PATH

### Calibración de zona Webcam

```bash
python tools/calibrate.py webcam --file videos/input/stream.mp4 --second 60
```

- Extrae un frame del segundo indicado
- Abre una ventana OpenCV para marcar la zona de la webcam con el ratón (clic y arrastra)
- Calcula y muestra los valores exactos para `config.json → layout`:

```
"webcam_w_ratio":        0.137,
"webcam_h_ratio":        0.331,
"webcam_x_offset":       22,
"webcam_y_center_ratio": 0.500,
```

### Medición del Border Ratio

```bash
# Frame con cámara visible
python tools/calibrate.py border --file stream.mp4 --second 120

# Comparar frame con cámara vs. sin cámara
python tools/calibrate.py border --file stream.mp4 --second 120 --second2 300
```

- Mide la fracción de píxeles del color del borde en la zona webcam configurada
- Muestra el valor medido vs. el `border_min_ratio` actual y sugiere ajustes
- Abre ventana OpenCV mostrando los píxeles detectados en magenta, la zona webcam en verde y la zona de borde en azul

---

## Publicación automática

### YouTube Shorts

**Requisitos:**
1. Crear proyecto en [Google Cloud Console](https://console.cloud.google.com) con **YouTube Data API v3** habilitada
2. Crear credenciales OAuth2 (tipo "Desktop") y descargar `client_secrets.json` en la raíz del proyecto
3. La primera ejecución del Paso 8 abre el navegador para autorización — genera `token.pickle` para ejecuciones futuras

### TikTok

**Requisitos:**
1. Tener `playwright` instalado (`playwright install`)
2. Exportar las cookies de sesión de TikTok en formato Netscape a `cookies.txt` en la raíz del proyecto (usar extensión de navegador como "Get cookies.txt LOCALLY")

### Programación horaria

En la UI (Pestaña 2), activar "Programar publicación" e indicar fecha y hora. El JSON de metadata guarda el campo `schedule_time` en ISO 8601. El Paso 8 lo envía a YouTube (`publishAt`, el vídeo se sube como privado y se publica a la hora indicada) o a Playwright para TikTok.

---

## Caché del pipeline

La caché evita reprocesar pasos costosos cuando el vídeo y la configuración no han cambiado.

### Estructura por vídeo

```
videos/cache/{video_stem}/
├── meta.json              # Fingerprints de config + flags is_driving por clip
├── transcription.json     # Lista de palabras con timestamps (Whisper)
├── segments.json          # Segmentos virales detectados (editable desde la UI)
└── face_{clip_stem}.json  # Datos de detección facial por clip
```

### Invalidación automática

La caché de cada paso se invalida si:
- El archivo de vídeo cambia (tamaño o fecha de modificación)
- Cambian los parámetros de config relevantes para ese paso (modelo Whisper, parámetros de detección, etc.)

### Gestión por línea de comandos

```bash
# Borrar caché de un vídeo antes de procesar
python main.py --file stream.mp4 --clear-cache

# Ignorar caché completamente en esta ejecución
python main.py --file stream.mp4 --no-cache
```

Desde la UI: **Pestaña 3 — Caché** para borrado granular o total.

---

## Notas de rendimiento

- El pipeline completo de un stream de 30 minutos tarda aproximadamente 8–15 minutos en una RTX 4080 Super
- La transcripción con Whisper `large-v3` es el paso más lento (~5–8 min para 30 min de vídeo); reducir a `medium` o `small` si la velocidad es prioritaria
- La diarización con Pyannote añade ~3–5 minutos adicionales; desactivar si no se necesitan colores por locutor (poner `transcriber.diarization_enabled: false` en config o en la Pestaña 5 — Configuración)
- El análisis multimodal (`ai_features.multimodal_video`) añade ~1–3 minutos extra (extracción de frames con FFmpeg + tokens adicionales de imagen); desactivar si se prioriza velocidad sobre calidad de detección visual
- Los Pasos 3–6 (extracción, composición, render) se benefician directamente de NVDEC + NVENC
- El modo `--max-step 2` permite revisar los clips en la UI antes de renderizar, evitando renders innecesarios
