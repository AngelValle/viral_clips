"""
ui.py
Interfaz Web para revisar cortes, elegir plataformas y lanzar tareas.
Ejecución: streamlit run ui.py
"""

import re as _re
import html as _html
import streamlit as st
import json
import datetime
import os
import subprocess
import tempfile
from pathlib import Path
from modules.scanner import scan_folder, load_config
from modules.cache import PipelineCache

st.set_page_config(page_title="Viral Clips AI - Dashboard", layout="wide", page_icon="🚀")
st.title("🎬 Viral Clips AI - Dashboard")

config = load_config("config.json")
input_dir = Path(config["paths"]["input_dir"])
videos = scan_folder(input_dir, config["paths"]["supported_formats"])

if not videos:
    st.warning(f"No hay vídeos en la carpeta de entrada: {input_dir}")
    st.stop()

tab_lanzador, tab_revisor, tab_cache, tab_tools, tab_config, tab_analytics = st.tabs([
    "🚀 1. Lanzador de Tareas",
    "✂️ 2. Revisión y Publicación",
    "🗑️ 3. Caché",
    "🔧 4. Herramientas",
    "⚙️ 5. Configuración",
    "📊 6. Analítica",
])

# ── Estado global ─────────────────────────────────────────────────────────────
if "global_log" not in st.session_state:
    st.session_state["global_log"] = []
if "running" not in st.session_state:
    st.session_state["running"] = False
if "progress_info" not in st.session_state:
    st.session_state["progress_info"] = {"step": 0, "max_step": 8, "video": "", "video_idx": 0, "total_videos": 0}

def _append_log(line: str):
    st.session_state["global_log"].append(line)
    if len(st.session_state["global_log"]) > 500:
        st.session_state["global_log"] = st.session_state["global_log"][-500:]

def _classify(line: str) -> str:
    l = line.lower()
    if any(x in l for x in ["error", "falló", "failed", "traceback", "exception", "❌"]):
        return "error"
    if any(x in l for x in ["warning", "warn", "⚠"]):
        return "warn"
    if any(x in l for x in ["✅", "completado", "generado", "✔", "éxito", "success"]):
        return "ok"
    if any(x in l for x in ["paso", "iniciando", "▶", "transcrib", "detección", "render", "cargando", "subiendo", "clip"]):
        return "info"
    return "default"

COLORS = {
    "error":   ("rgba(255,60,60,0.08)",  "#ff6b6b", "#ff4444"),
    "warn":    ("rgba(255,180,0,0.06)",  "#ffd166", "#e8a202"),
    "ok":      ("rgba(40,200,100,0.06)", "#6bffb8", "#28c840"),
    "info":    ("transparent",           "#82aaff", "#4466cc"),
    "default": ("transparent",           "#a8aab8", "transparent"),
}

def _render_log_md(lines):
    rows_html = ""
    # Orden inverso: con flex column-reverse el primero del HTML queda abajo visualmente,
    # por lo que el scroll empieza siempre al fondo sin necesitar JavaScript.
    for raw in reversed(lines[-200:] if lines else []):
        kind = _classify(raw)
        safe = _html.escape(_re.sub(r'\x1b\[[0-9;]*m', '', raw))
        bg, fg, border = COLORS[kind]
        border_css = f"border-left:2px solid {border};" if border != "transparent" else ""
        rows_html += (
            f'<div style="padding:2px 14px;font-size:12.5px;line-height:1.75;'
            f'white-space:pre-wrap;word-break:break-all;background:{bg};'
            f'{border_css}color:{fg};font-family:Consolas,\'Fira Mono\',monospace">'
            f'{safe}</div>'
        )
    if not rows_html:
        rows_html = '<div style="padding:8px 14px;color:#555;font-family:monospace;font-size:12px">Sin actividad todavía...</div>'

    return f"""<div style="background:#0f1117;border-radius:10px;border:1px solid #2a2d3a;overflow:hidden">
<div style="display:flex;align-items:center;gap:8px;padding:10px 16px;background:#181b27;border-bottom:1px solid #2a2d3a">
<span style="width:12px;height:12px;border-radius:50%;background:#ff5f57;display:inline-block"></span>
<span style="width:12px;height:12px;border-radius:50%;background:#ffbd2e;display:inline-block"></span>
<span style="width:12px;height:12px;border-radius:50%;background:#28c840;display:inline-block"></span>
<span style="color:#8b8fa8;font-size:12px;margin-left:4px;letter-spacing:.05em;font-family:monospace">PIPELINE LOG — viral_clips</span>
</div>
<div id="pipeline-log-scroll" style="max-height:380px;overflow-y:auto;padding:8px 0;display:flex;flex-direction:column-reverse">{rows_html}</div>
</div>"""

_PASO_NAMES = {
    1: "Transcripción Whisper",
    2: "Detección Viral",
    3: "Extracción de Segmentos",
    4: "Detección Facial",
    5: "Composición Visual",
    6: "Censura + Subtítulos + Render",
    7: "Metadatos con IA",
    8: "Auto-Publicación",
}

def _update_progress(line: str):
    m = _re.search(r'PASO\s+(\d+)/(\d+)', line)
    if m:
        st.session_state["progress_info"]["step"]     = int(m.group(1))
        st.session_state["progress_info"]["max_step"] = int(m.group(2))

def _render_progress_html() -> str:
    info      = st.session_state.get("progress_info", {})
    step      = info.get("step", 0)
    max_step  = info.get("max_step", 8)
    video     = info.get("video", "")
    video_idx = info.get("video_idx", 0)
    total     = info.get("total_videos", 0)

    if step == 0 and not st.session_state.get("running"):
        return ""

    pct        = step / max_step if max_step > 0 else 0
    bar_w      = int(pct * 100)
    paso_label = _PASO_NAMES.get(step, f"Paso {step}") if step > 0 else "Esperando..."
    video_label = (f"Vídeo {video_idx}/{total}: {_html.escape(video)}"
                   if total > 0 else _html.escape(video))

    return (
        f'<div style="background:#0f1117;border-radius:8px;border:1px solid #2a2d3a;'
        f'padding:10px 16px;margin-top:4px">'
        f'<div style="display:flex;justify-content:space-between;margin-bottom:7px">'
        f'<span style="color:#82aaff;font-size:12px;font-family:monospace">{video_label}</span>'
        f'<span style="color:#8b8fa8;font-size:12px;font-family:monospace">'
        f'PASO {step}/{max_step} — {paso_label}</span>'
        f'</div>'
        f'<div style="background:#1e2030;border-radius:4px;height:6px;overflow:hidden">'
        f'<div style="background:linear-gradient(90deg,#4466cc,#82aaff);'
        f'width:{bar_w}%;height:100%;border-radius:4px"></div>'
        f'</div>'
        f'</div>'
    )

def _run_with_log(cmd: list, log_placeholder, progress_placeholder=None):
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True, encoding="utf-8", errors="replace", env=env) as proc:
        for line in proc.stdout:
            _append_log(line.rstrip())
            _update_progress(line.rstrip())
            log_placeholder.markdown(_render_log_md(st.session_state["global_log"]), unsafe_allow_html=True)
            if progress_placeholder is not None:
                progress_placeholder.markdown(_render_progress_html(), unsafe_allow_html=True)
    return proc.returncode

# ── Log único — debe estar definido antes de cualquier tab que lo use ─────────
st.divider()
_log_col, _btn_col = st.columns([6, 1])
with _log_col:
    st.markdown("#### Pipeline Log")
with _btn_col:
    if st.button("Limpiar", key="clear_log"):
        st.session_state["global_log"] = []
        st.rerun()
LOG_PLACEHOLDER = st.empty()
LOG_PLACEHOLDER.markdown(_render_log_md(st.session_state["global_log"]), unsafe_allow_html=True)
PROGRESS_PLACEHOLDER = st.empty()
PROGRESS_PLACEHOLDER.markdown(_render_progress_html(), unsafe_allow_html=True)
st.divider()

# ==========================================
# PESTAÑA 3: GESTIÓN DE CACHÉ
# ==========================================
with tab_cache:
    import shutil as _shutil
    import json as _json_cache
    cache_dir = Path(config["paths"].get("cache_dir", "videos/cache"))

    st.markdown("### Gestión de Caché del Pipeline")
    st.caption(f"Directorio: `{cache_dir.resolve()}`")

    def _cache_size(path: Path) -> int:
        if not path.exists(): return 0
        return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())

    def _fmt_size(b: int) -> str:
        if b < 1024: return f"{b} B"
        if b < 1024**2: return f"{b/1024:.1f} KB"
        return f"{b/1024**2:.1f} MB"

    total_size = _cache_size(cache_dir)
    st.info(f"Tamaño total de caché: **{_fmt_size(total_size)}**")
    st.divider()

    cache_entries = sorted(cache_dir.iterdir()) if cache_dir.exists() else []
    video_caches  = [e for e in cache_entries if e.is_dir()]

    if not video_caches:
        st.success("La caché está vacía.")
    else:
        for vc in video_caches:
            has_transcription = (vc / "transcription.json").exists()
            has_segments      = (vc / "segments.json").exists()
            face_files        = sorted(vc.glob("face_*.json"))
            total_vc_size     = _cache_size(vc)

            tags = []
            if has_transcription: tags.append("📝")
            if has_segments:      tags.append("✂️")
            if face_files:        tags.append("🎭")
            tag_str = "  ".join(tags) if tags else "vacío"

            with st.expander(f"**{vc.name}**  {tag_str}  —  {_fmt_size(total_vc_size)}"):
                if not has_transcription and not has_segments and not face_files:
                    st.caption("Carpeta vacía.")
                else:
                    parts_to_delete = []

                    if has_transcription:
                        sz = (vc / "transcription.json").stat().st_size
                        if st.checkbox(f"📝 Transcripción  ({_fmt_size(sz)})", key=f"del_tr_{vc.name}"):
                            parts_to_delete.append("transcription")

                    if has_segments:
                        sz = (vc / "segments.json").stat().st_size
                        if st.checkbox(f"✂️ Segmentos virales  ({_fmt_size(sz)})", key=f"del_seg_{vc.name}"):
                            parts_to_delete.append("segments")

                    if face_files:
                        face_sz = sum(f.stat().st_size for f in face_files)
                        if st.checkbox(
                            f"🎭 Datos faciales  ({len(face_files)} clip{'s' if len(face_files)!=1 else ''}, {_fmt_size(face_sz)})",
                            key=f"del_face_{vc.name}",
                        ):
                            parts_to_delete.append("face")

                    st.write("")
                    col1, col2 = st.columns(2)

                    with col1:
                        if st.button("🗑️ Borrar seleccionados", key=f"btn_del_{vc.name}",
                                     disabled=not parts_to_delete, width="stretch",
                                     type="primary"):
                            deleted = []
                            if "transcription" in parts_to_delete:
                                (vc / "transcription.json").unlink(missing_ok=True)
                                deleted.append("transcripción")
                            if "segments" in parts_to_delete:
                                (vc / "segments.json").unlink(missing_ok=True)
                                deleted.append("segmentos")
                            if "face" in parts_to_delete:
                                for ff in face_files:
                                    ff.unlink(missing_ok=True)
                                # Limpiar entradas de cara en meta.json
                                meta_p = vc / "meta.json"
                                if meta_p.exists():
                                    try:
                                        meta = _json_cache.loads(meta_p.read_text(encoding="utf-8"))
                                        meta = {k: v for k, v in meta.items()
                                                if not k.startswith("face_cfg_fp_")
                                                and not k.startswith("is_driving_")}
                                        meta_p.write_text(
                                            _json_cache.dumps(meta, ensure_ascii=False, indent=2),
                                            encoding="utf-8",
                                        )
                                    except Exception:
                                        pass
                                deleted.append("datos faciales")
                            _append_log(f"🗑️ {vc.name}: borrado {', '.join(deleted)}")
                            st.success(f"✅ Borrado: {', '.join(deleted)}")
                            st.rerun()

                    with col2:
                        if st.button("💣 Borrar proyecto completo", key=f"btn_del_all_{vc.name}",
                                     type="secondary", width="stretch"):
                            _shutil.rmtree(vc, ignore_errors=True)
                            _append_log(f"💣 Caché eliminada: {vc.name}")
                            st.success(f"✅ Caché de {vc.name} eliminada.")
                            st.rerun()

        st.divider()
        if st.button("💣 Limpiar TODO", type="secondary", width="stretch"):
            _shutil.rmtree(cache_dir, ignore_errors=True)
            cache_dir.mkdir(parents=True, exist_ok=True)
            _append_log("💣 Caché completa eliminada.")
            st.success("✅ Caché completa eliminada.")
            st.rerun()


# ==========================================
# PESTAÑA 4: HERRAMIENTAS DE CALIBRACIÓN
# ==========================================
with tab_tools:
    st.markdown("### 🔧 Herramientas de Calibración")

    # ── GPU ───────────────────────────────────────────────────────────────────
    st.markdown("#### 🖥️ Diagnóstico GPU")
    st.caption("Comprueba drivers NVIDIA, PyTorch CUDA, FFmpeg NVENC y el encoder h264_nvenc.")
    if st.button("▶ Ejecutar diagnóstico GPU", key="btn_gpu", width="stretch"):
        _append_log("▶ Iniciando diagnóstico GPU...")
        rc = _run_with_log(["python", "-u", "tools/calibrate.py", "gpu"], LOG_PLACEHOLDER)
        if rc == 0:
            st.success("✅ Diagnóstico completado.")
        else:
            st.error(f"❌ Error (código {rc})")

    st.divider()

    # ── WEBCAM ────────────────────────────────────────────────────────────────
    st.markdown("#### 📷 Seleccionar zona Webcam")
    st.caption(
        "Abre una ventana OpenCV para marcar visualmente la zona de la webcam. "
        "Copia los valores que aparezcan en el log a `config.json → layout`."
    )
    col_wc1, col_wc2 = st.columns([4, 1])
    with col_wc1:
        wc_file = st.selectbox("Vídeo de referencia:", [str(v) for v in videos], key="wc_file")
    with col_wc2:
        wc_second = st.number_input("Segundo", min_value=0, value=60, step=10, key="wc_second")

    if st.button("▶ Abrir selector de webcam", key="btn_webcam", width="stretch"):
        _append_log(f"▶ Webcam calibration: {Path(wc_file).name}  s={wc_second}")
        rc = _run_with_log(
            ["python", "-u", "tools/calibrate.py", "webcam",
             "--file", wc_file, "--second", str(wc_second)],
            LOG_PLACEHOLDER,
        )
        if rc == 0:
            st.success("✅ Copia los valores del log en config.json → layout.")
        else:
            st.error(f"❌ Error (código {rc})")

    st.divider()

    # ── BORDER ────────────────────────────────────────────────────────────────
    st.markdown("#### 🟣 Medir Border Ratio")
    st.caption(
        "Mide el ratio del borde morado de la webcam para calibrar `border_min_ratio` "
        "en `config.json → face_detection`."
    )
    col_b1, col_b2, col_b3 = st.columns([4, 1, 1])
    with col_b1:
        bd_file = st.selectbox("Vídeo de referencia:", [str(v) for v in videos], key="bd_file")
    with col_b2:
        bd_second = st.number_input("Segundo (con cámara)", min_value=0, value=60, step=10, key="bd_second")
    with col_b3:
        bd_second2_raw = st.number_input("Segundo 2 (sin cámara)", min_value=0, value=0, step=10, key="bd_second2",
                                          help="Opcional. Deja en 0 para omitir.")

    if st.button("▶ Medir border ratio", key="btn_border", width="stretch"):
        _append_log(f"▶ Border calibration: {Path(bd_file).name}  s={bd_second}")
        cmd = ["python", "-u", "tools/calibrate.py", "border",
               "--file", bd_file, "--second", str(bd_second)]
        if bd_second2_raw > 0:
            cmd += ["--second2", str(bd_second2_raw)]
        rc = _run_with_log(cmd, LOG_PLACEHOLDER)
        if rc == 0:
            st.success("✅ Consulta el log para ver el `border_min_ratio` medido.")
        else:
            st.error(f"❌ Error (código {rc})")


# ==========================================
# PESTAÑA 1: LANZADOR MULTI-VÍDEO
# ==========================================
with tab_lanzador:
    st.markdown("### Selecciona qué hacer con cada vídeo de la carpeta:")

    opciones_pasos = {
        1: "1. Solo Transcribir (Whisper)",
        2: "2. Detección Viral (Recomendado para revisar luego)",
        3: "3. Extraer Segmentos Crudos",
        5: "5. Composición Visual",
        7: "7. Render Final + Metadatos",
        8: "8. Pipeline Completo (Hasta Auto-Publicación)"
    }

    tareas_a_ejecutar = {}

    for v in videos:
        col1, col2, col3 = st.columns([0.5, 3, 3])
        with col1:
            marcado = st.checkbox("Procesar", key=f"run_{v.name}")
        with col2:
            st.markdown(f"**📄 {v.name}**")
            st.caption(f"{(v.stat().st_size / (1024*1024)):.1f} MB")
        with col3:
            paso_elegido = st.selectbox("Límite de ejecución:", options=list(opciones_pasos.keys()),
                                        format_func=lambda x: opciones_pasos[x], index=1, key=f"step_{v.name}")
        if marcado:
            tareas_a_ejecutar[v] = paso_elegido
        st.divider()

    if st.session_state["running"]:
        st.warning("⏳ Ya hay un proceso en ejecución. Espera a que termine.")
    elif st.button("▶️ Lanzar Procesamiento Seleccionado", type="primary", width="stretch"):
        if not tareas_a_ejecutar:
            st.error("No has marcado ningún vídeo para procesar.")
        else:
            st.session_state["running"] = True
            _total = len(tareas_a_ejecutar)
            for _idx, (video_file, max_step) in enumerate(tareas_a_ejecutar.items(), 1):
                st.session_state["progress_info"] = {
                    "step": 0, "max_step": max_step,
                    "video": video_file.name,
                    "video_idx": _idx, "total_videos": _total,
                }
                _append_log(f"▶ Iniciando: {video_file.name} (paso {max_step})")
                cmd = ["python", "-u", "main.py", "--file", str(video_file.resolve()), "--max-step", str(max_step)]
                rc = _run_with_log(cmd, LOG_PLACEHOLDER, PROGRESS_PLACEHOLDER)
                if rc != 0:
                    _append_log(f"❌ Falló: {video_file.name} (código {rc})")
                    st.error(f"❌ Error en {video_file.name} (código {rc})")
                else:
                    _append_log(f"✅ Completado: {video_file.name}")
                    st.success(f"✅ {video_file.name} completado.")
            st.session_state["running"] = False
            st.session_state["progress_info"] = {"step": 0, "max_step": 8, "video": "", "video_idx": 0, "total_videos": 0}
            PROGRESS_PLACEHOLDER.empty()
            st.success("🎉 ¡Todas las tareas han finalizado!")

# ==========================================
# PESTAÑA 2: REVISIÓN DE CORTES
# ==========================================
with tab_revisor:
    videos_con_cache = [v for v in videos if PipelineCache(v, config).get_segments() is not None]

    if not videos_con_cache:
        st.info("No hay vídeos listos para revisar. Lanza un vídeo hasta el **Paso 2** en la pestaña anterior.")
    else:
        video_names = [v.name for v in videos_con_cache]
        selected_video_name = st.selectbox("Selecciona un directo para revisar:", video_names)
        selected_video = next(v for v in videos_con_cache if v.name == selected_video_name)

        cache = PipelineCache(selected_video, config)
        segments = cache.get_segments()
        local_tz = datetime.datetime.now().astimezone().tzinfo

        # ── Duración total del vídeo (cacheada en session_state) ─────────────
        _vdur_key = f"vid_dur_{selected_video.stem}"
        if _vdur_key not in st.session_state:
            try:
                _vr = subprocess.run(
                    ["ffprobe", "-v", "quiet", "-print_format", "json",
                     "-show_format", str(selected_video)],
                    capture_output=True, text=True,
                )
                st.session_state[_vdur_key] = float(json.loads(_vr.stdout)["format"]["duration"])
            except Exception:
                st.session_state[_vdur_key] = max(
                    (float(s["end"]) for s in segments), default=300.0
                ) + 30.0
        vid_dur = float(st.session_state[_vdur_key])

        # ── Inicializar fragmentos editables en session_state ─────────────────
        for _i, _seg in enumerate(segments):
            _fk = f"clip_frags_{selected_video.stem}_{_i}"
            if _fk not in st.session_state:
                _raw = _seg.get("fragments")
                if _raw:
                    st.session_state[_fk] = [dict(f) for f in _raw]
                else:
                    st.session_state[_fk] = [{"start": float(_seg["start"]), "end": float(_seg["end"])}]

        # ── Un bloque por clip: preview izquierda + controles derecha ──────────
        # Los botones van fuera del form; el form solo recoge el estado al guardar.

        updated_segments = []  # se rellena en el form de abajo

        for i, seg in enumerate(segments):
            st.markdown(f"### ✂️ Clip {i+1}: {seg.get('gemini_desc', 'Sin descripción')}")
            st.caption(f"Score Viral: {seg.get('score', 0):.2f}")

            col_vid, col_edit = st.columns([2, 1.5])

            # ── Columna izquierda: vista previa ──────────────────────────────
            with col_vid:
                preview_path = Path(tempfile.gettempdir()) / f"preview_{selected_video.stem}_{i}.mp4"
                preview_key = f"preview_ready_{selected_video.stem}_{i}"

                if st.button(f"👁️ Vista previa — Clip {i+1}", key=f"ver_{i}"):
                    frags = seg.get("fragments")
                    if frags and len(frags) > 1:
                        # Multi-fragmento: concatenar con FFmpeg concat demuxer
                        import tempfile as _tf
                        tmp_dir = Path(_tf.mkdtemp())
                        parts = []
                        for fi, frag in enumerate(frags):
                            part = tmp_dir / f"part_{fi}.mp4"
                            dur = float(frag["end"]) - float(frag["start"])
                            r = subprocess.run([
                                "ffmpeg", "-y", "-hwaccel", "cuda",
                                "-ss", str(frag["start"]), "-i", str(selected_video),
                                "-t", str(dur), "-vf", "scale=-2:480",
                                "-c:v", "h264_nvenc", "-preset", "p1", "-cq", "35",
                                "-c:a", "aac", str(part)
                            ], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
                            if r.returncode != 0:
                                subprocess.run([
                                    "ffmpeg", "-y", "-ss", str(frag["start"]), "-i", str(selected_video),
                                    "-t", str(dur), "-vf", "scale=-2:480",
                                    "-c:v", "libx264", "-preset", "ultrafast", "-crf", "35",
                                    "-c:a", "aac", str(part)
                                ], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
                            parts.append(part)
                        list_file = tmp_dir / "list.txt"
                        list_file.write_text("\n".join(f"file '{p}'" for p in parts), encoding="utf-8")
                        subprocess.run([
                            "ffmpeg", "-y", "-f", "concat", "-safe", "0",
                            "-i", str(list_file), "-c", "copy", str(preview_path)
                        ], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
                    else:
                        dur = float(seg["end"]) - float(seg["start"])
                        r = subprocess.run([
                            "ffmpeg", "-y", "-hwaccel", "cuda",
                            "-ss", str(seg["start"]), "-i", str(selected_video),
                            "-t", str(dur), "-vf", "scale=-2:480",
                            "-c:v", "h264_nvenc", "-preset", "p1", "-cq", "35",
                            "-c:a", "aac", str(preview_path)
                        ], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
                        if r.returncode != 0:
                            subprocess.run([
                                "ffmpeg", "-y", "-ss", str(seg["start"]), "-i", str(selected_video),
                                "-t", str(dur), "-vf", "scale=-2:480",
                                "-c:v", "libx264", "-preset", "ultrafast", "-crf", "35",
                                "-c:a", "aac", str(preview_path)
                            ], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
                    st.session_state[preview_key] = True
                    st.rerun()

                if st.session_state.get(preview_key) and preview_path.exists():
                    with open(preview_path, "rb") as f:
                        st.video(f.read(), format="video/mp4")

            # ── Columna derecha: fragmentos editables + publicación ───────────
            with col_edit:
                _fk = f"clip_frags_{selected_video.stem}_{i}"
                _cur_frags = st.session_state[_fk]

                st.markdown("#### ✂️ Fragmentos")
                _total_s = sum(f["end"] - f["start"] for f in _cur_frags)
                st.caption(f"{len(_cur_frags)} fragmento(s) — {_total_s:.1f}s total")

                for _fi, _frag in enumerate(_cur_frags):
                    _cf, _cd = st.columns([5, 1])
                    _cf.markdown(f"**Frag {_fi+1}:** `{_frag['start']:.1f}s → {_frag['end']:.1f}s`")
                    if _cd.button("✕", key=f"del_{i}_{_fi}", help="Eliminar fragmento",
                                  disabled=len(_cur_frags) <= 1):
                        st.session_state[_fk].pop(_fi)
                        st.rerun()

                if st.button("➕ Añadir fragmento", key=f"add_{i}"):
                    _last = _cur_frags[-1]
                    _ns = round(min(_last["end"] + 1.0, max(0.0, vid_dur - 2.0)), 1)
                    _ne = round(min(_ns + 5.0, vid_dur), 1)
                    st.session_state[_fk].append({"start": _ns, "end": _ne})
                    st.rerun()

                st.markdown("#### 🚀 Publicación")
                st.caption("Marca las plataformas en el formulario de abajo.")

            st.divider()

        # ── Formulario: solo recoge aprobación, tiempos y publicación ─────────
        with st.form("review_form"):
            updated_segments = []

            for i, seg in enumerate(segments):
                _fk = f"clip_frags_{selected_video.stem}_{i}"
                _form_frags = st.session_state.get(
                    _fk,
                    [{"start": float(seg["start"]), "end": float(seg["end"])}],
                )

                st.markdown(f"**Clip {i+1}** — {seg.get('gemini_desc', '')}")
                aprobado = st.checkbox("✅ Aprobar", value=True, key=f"chk_{i}")

                new_frags = []
                for fi, frag in enumerate(_form_frags):
                    fs, fe = st.slider(
                        f"Fragmento {fi+1}:",
                        min_value=0.0, max_value=float(vid_dur),
                        value=(float(frag["start"]), float(frag["end"])),
                        step=0.1, key=f"frag_{i}_{fi}"
                    )
                    new_frags.append({"start": fs, "end": fe})
                new_start = new_frags[0]["start"]
                new_end = new_frags[-1]["end"]

                col_yt, col_tk = st.columns(2)
                pub_yt = col_yt.checkbox("▶ YouTube Shorts", value=False, key=f"yt_{i}")
                pub_tk = col_tk.checkbox("♪ TikTok", value=False, key=f"tk_{i}")

                schedule_iso = None
                if pub_yt or pub_tk:
                    programar = st.checkbox("📅 Programar publicación", key=f"prog_{i}")
                    if programar:
                        col_d, col_t = st.columns(2)
                        d = col_d.date_input("Fecha", min_value=datetime.date.today(), key=f"date_{i}")
                        t = col_t.time_input("Hora", value=datetime.time(18, 0), key=f"time_{i}")
                        dt_aware = datetime.datetime.combine(d, t).replace(tzinfo=local_tz)
                        schedule_iso = dt_aware.isoformat()
                        st.info(f"Se publicará el: {dt_aware.strftime('%d/%m/%Y a las %H:%M')}")

                if aprobado:
                    updated_seg = seg.copy()
                    updated_seg["start"] = new_start
                    updated_seg["end"] = new_end
                    updated_seg["fragments"] = new_frags
                    updated_seg["publish_yt"] = pub_yt
                    updated_seg["publish_tk"] = pub_tk
                    updated_seg["schedule_time"] = schedule_iso
                    updated_segments.append(updated_seg)

                st.divider()

            submitted = st.form_submit_button("💾 Guardar Ajustes de Edición", type="primary")

            if submitted:
                for i in range(len(segments)):
                    p = Path(tempfile.gettempdir()) / f"preview_{selected_video.stem}_{i}.mp4"
                    if p.exists(): p.unlink()
                    st.session_state.pop(f"preview_ready_{selected_video.stem}_{i}", None)
                    # Reinicializar fragmentos desde los datos guardados en el próximo render
                    st.session_state.pop(f"clip_frags_{selected_video.stem}_{i}", None)
                if not updated_segments:
                    st.error("No has aprobado ningún clip.")
                else:
                    cache.save_segments(updated_segments)
                    st.session_state["ready_to_render"] = selected_video.resolve()
                    st.success("¡Cortes y programación guardados!")

        if st.session_state.get("ready_to_render") == selected_video.resolve():
            st.markdown("### 🔥 Paso Final")
            if st.session_state["running"]:
                st.warning("⏳ Ya hay un proceso en ejecución. Espera a que termine.")
            elif st.button("🎬 Renderizar y Publicar Clips Aprobados (Pasos 3 al 8)", type="primary", width="stretch"):
                st.session_state["running"] = True
                _append_log(f"▶ Renderizando: {selected_video.name}")
                cmd = ["python", "-u", "main.py", "--file", str(selected_video), "--max-step", "8"]
                rc = _run_with_log(cmd, LOG_PLACEHOLDER)
                st.session_state["running"] = False
                if rc != 0:
                    _append_log(f"❌ Render fallido: {selected_video.name} (código {rc})")
                    st.error(f"❌ Error en el render (código {rc})")
                else:
                    _append_log(f"✅ Render completado: {selected_video.name}")
                    st.success("🎉 ¡Todos los clips han sido renderizados y gestionados!")


# ==========================================
# PESTAÑA 5: CONFIGURACIÓN
# ==========================================
with tab_config:
    import copy as _copy

    CONFIG_PATH = Path("config.json")

    def _load_cfg() -> dict:
        return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))

    def _save_cfg(cfg: dict):
        CONFIG_PATH.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")

    cfg = _load_cfg()

    st.markdown("### ⚙️ Configuración del Pipeline")
    st.caption("Los cambios se aplican al guardar. El pipeline usará la nueva config en la siguiente ejecución.")

    # ── Streamer ────────────────────────────────────────────────────────────────
    with st.expander("🎮 Información del Streamer", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            streamer_name = st.text_input(
                "Nombre del streamer",
                value=cfg.get("claude", {}).get("streamer_name", ""),
                help="Se usa en subtítulos, metadatos y títulos generados por IA.",
            )
        with col2:
            game_name = st.text_input(
                "Nombre del juego",
                value=cfg.get("claude", {}).get("game_name", ""),
                help="Se incluye en los títulos y descripciones de los clips.",
            )
        with col3:
            content_type = st.selectbox(
                "Tipo de contenido",
                options=["gaming", "irl", "just_chatting", "sports", "music"],
                index=["gaming", "irl", "just_chatting", "sports", "music"].index(
                    cfg.get("claude", {}).get("content_type", "gaming")
                ),
            )

    # ── Módulos opcionales ──────────────────────────────────────────────────────
    with st.expander("🔀 Módulos Opcionales del Pipeline", expanded=True):
        st.caption("Activa o desactiva partes del pipeline para ajustar velocidad y resultados.")
        col1, col2 = st.columns(2)
        with col1:
            face_enabled = st.toggle(
                "🎭 Detección de cámara/cara",
                value=cfg.get("face_detection", {}).get("enabled", True),
                help="Si está deshabilitado, el clip se renderiza en modo pantalla completa sin overlay de webcam.",
            )
            subtitles_enabled = st.toggle(
                "💬 Subtítulos",
                value=cfg.get("subtitles", {}).get("enabled", True),
                help="Genera y embede subtítulos estilo karaoke TikTok en el clip final.",
            )
        with col2:
            detect_driving = st.toggle(
                "🚗 Detección de modo conducción",
                value=cfg.get("layout", {}).get("detect_driving", True),
                help="Detecta automáticamente escenas de conducción para ajustar el layout del HUD.",
            )
            diarization_enabled = st.toggle(
                "🎙️ Diarización de locutores",
                value=cfg.get("transcriber", {}).get("diarization_enabled", True),
                help="Identifica a cada locutor con Pyannote. Requiere HF_TOKEN en el entorno.",
            )

    # ── Transcripción ───────────────────────────────────────────────────────────
    with st.expander("🎤 Transcripción — Whisper"):
        col1, col2, col3 = st.columns(3)
        with col1:
            whisper_model = st.selectbox(
                "Modelo Whisper",
                options=["tiny", "base", "small", "medium", "large-v2", "large-v3"],
                index=["tiny", "base", "small", "medium", "large-v2", "large-v3"].index(
                    cfg.get("whisper", {}).get("model", "large-v3")
                ),
                help="Modelos más grandes = mayor precisión pero más lento y más VRAM.",
            )
        with col2:
            whisper_language = st.text_input(
                "Idioma (código ISO)",
                value=cfg.get("whisper", {}).get("language", "es"),
                help="Ej: 'es' español, 'en' inglés, 'auto' para detección automática.",
            )
        with col3:
            compute_options = ["float16", "float32", "int8", "int8_float16"]
            compute_val = cfg.get("whisper", {}).get("compute_type", "float16")
            whisper_compute = st.selectbox(
                "Tipo de cómputo",
                options=compute_options,
                index=compute_options.index(compute_val) if compute_val in compute_options else 0,
                help="float16 es óptimo para GPU NVIDIA. int8 para CPU.",
            )

    # ── Detección viral ─────────────────────────────────────────────────────────
    with st.expander("🔥 Detección de Momentos Virales"):
        vd = cfg.get("viral_detection", {})
        col1, col2, col3 = st.columns(3)
        with col1:
            min_clip_dur = st.number_input(
                "Duración mínima del clip (s)",
                min_value=1, max_value=120, value=int(vd.get("min_clip_duration", 10)),
            )
            pre_buffer = st.number_input(
                "Buffer previo al momento viral (s)",
                min_value=0, max_value=30, value=int(vd.get("pre_buffer_seconds", 3)),
            )
            skip_intro = st.number_input(
                "Saltar intro del vídeo (s)",
                min_value=0, max_value=300, value=int(vd.get("skip_intro_sec", 28)),
                help="Ignora los primeros N segundos del stream.",
            )
        with col2:
            max_clip_dur = st.number_input(
                "Duración máxima del clip (s)",
                min_value=5, max_value=300, value=int(vd.get("max_clip_duration", 60)),
            )
            post_buffer = st.number_input(
                "Buffer posterior al momento viral (s)",
                min_value=0, max_value=30, value=int(vd.get("post_buffer_seconds", 2)),
            )
            skip_outro = st.number_input(
                "Saltar outro del vídeo (s)",
                min_value=0, max_value=300, value=int(vd.get("skip_outro_sec", 28)),
                help="Ignora los últimos N segundos del stream.",
            )
        with col3:
            top_n_clips = st.number_input(
                "Número máximo de clips",
                min_value=1, max_value=9999, value=int(vd.get("top_n_clips", 999)),
                help="Limita cuántos clips se generan por vídeo.",
            )

        viral_keywords_raw = st.text_area(
            "Keywords virales (una por línea)",
            value="\n".join(vd.get("viral_keywords", [])),
            height=120,
            help="Palabras que indican un momento viral en la transcripción.",
        )

    # ── Subtítulos ──────────────────────────────────────────────────────────────
    with st.expander("💬 Subtítulos"):
        sub = cfg.get("subtitles", {})
        col1, col2, col3 = st.columns(3)
        with col1:
            sub_font_size = st.number_input(
                "Tamaño de fuente (px)", min_value=20, max_value=200,
                value=int(sub.get("font_size", 72)),
            )
            sub_outline = st.number_input(
                "Grosor del outline (px)", min_value=0, max_value=20,
                value=int(sub.get("outline_width", 4)),
            )
        with col2:
            sub_pos_y = st.slider(
                "Posición vertical (ratio 0-1)",
                min_value=0.0, max_value=1.0, step=0.01,
                value=float(sub.get("position_y_ratio", 0.82)),
                help="0 = arriba, 1 = abajo. 0.82 = zona inferior TikTok.",
            )
            sub_max_width = st.number_input(
                "Ancho máximo de línea (chars)", min_value=5, max_value=60,
                value=int(sub.get("max_line_width", 20)),
            )
        with col3:
            sub_font_name = st.text_input(
                "Fuente",
                value=sub.get("font_name", "Showcard Gothic"),
                help="Nombre exacto de la fuente instalada en Windows.",
            )
            sub_offset = st.number_input(
                "Offset manual (s)",
                min_value=-2.0, max_value=2.0, step=0.05,
                value=float(sub.get("manual_offset_sec", 0.4)),
                help="Ajusta el retardo de sincronía de los subtítulos.",
            )

    # ── Censura ─────────────────────────────────────────────────────────────────
    with st.expander("🤬 Censura de Audio"):
        cens = cfg.get("censorship", {})
        col1, col2 = st.columns(2)
        with col1:
            _cens_options = ["tiktok", "youtube", "instagram", "twitch", "desactivado"]
            _cens_current = cens.get("mode", "tiktok")
            cens_mode = st.selectbox(
                "Perfil de censura",
                options=_cens_options,
                index=_cens_options.index(_cens_current) if _cens_current in _cens_options else 0,
                help="Determina el nivel de agresividad del filtro de palabras. 'desactivado' omite la censura por completo.",
            )
            beep_freq = st.number_input(
                "Frecuencia del bip (Hz)", min_value=200, max_value=5000, step=50,
                value=int(cens.get("beep_frequency_hz", 1000)),
            )
        with col2:
            custom_words_raw = st.text_area(
                "Palabras censuradas adicionales (una por línea)",
                value="\n".join(cens.get("custom_words", [])),
                height=100,
                help="Se añaden al wordlist del perfil seleccionado.",
            )

    # ── IA / Gemini ─────────────────────────────────────────────────────────────
    with st.expander("🤖 IA — Gemini"):
        gem = cfg.get("gemini", {})
        ai  = cfg.get("ai_features", {})
        col1, col2 = st.columns(2)
        with col1:
            gemini_model = st.text_input(
                "Modelo Gemini",
                value=gem.get("model", "gemini-3.1-flash-lite-preview"),
                help="Nombre completo del modelo Gemini a usar para detección viral y metadatos.",
            )
        with col2:
            multimodal_video = st.toggle(
                "Análisis multimodal de vídeo",
                value=bool(ai.get("multimodal_video", False)),
                help=(
                    "Envía frames del stream a Gemini junto a la transcripción "
                    "para detectar momentos virales también por contexto visual. "
                    "Compatible con el tier gratuito. Añade ~1-3 min extra de procesamiento."
                ),
            )
            multimodal_interval = st.number_input(
                "Intervalo entre frames (s)",
                min_value=5,
                max_value=120,
                step=5,
                value=int(ai.get("multimodal_interval_sec", 10)),
                disabled=not multimodal_video,
                help="Cada cuántos segundos del stream se extrae un frame. Menos segundos = más contexto visual pero más tokens.",
            )

    # ── GPU / Hardware ───────────────────────────────────────────────────────────
    with st.expander("🖥️ GPU / Hardware"):
        gpu = cfg.get("gpu", {})
        force_cpu = st.toggle(
            "Forzar CPU (deshabilitar GPU)",
            value=bool(gpu.get("force_cpu", False)),
            help="Desactiva CUDA. Útil para debug o si la GPU no está disponible.",
        )

    # ── Salida ───────────────────────────────────────────────────────────────────
    with st.expander("📦 Salida de Vídeo"):
        out = cfg.get("output", {})
        col1, col2 = st.columns(2)
        with col1:
            output_fps = st.number_input(
                "FPS de salida", min_value=24, max_value=120, step=1,
                value=int(out.get("fps", 60)),
            )
            _orient_options = ["vertical", "horizontal"]
            _orient_current = out.get("orientation", "vertical")
            output_orientation = st.selectbox(
                "Formato",
                options=_orient_options,
                index=_orient_options.index(_orient_current) if _orient_current in _orient_options else 0,
                format_func=lambda x: "Vertical 9:16" if x == "vertical" else "Horizontal 16:9",
                help="Vertical 9:16 aplica composición con cámara. Horizontal 16:9 conserva el vídeo original sin reencuadre.",
            )
        with col2:
            naming_pattern = st.text_input(
                "Patrón de nombre de archivo",
                value=out.get("naming_pattern", "{source_name}_clip_{n}.mp4"),
                help="Variables disponibles: {source_name}, {n}",
            )

    # ── Guardar ──────────────────────────────────────────────────────────────────
    st.divider()
    if st.button("💾 Guardar Configuración", type="primary", width="stretch"):
        cfg_new = _copy.deepcopy(cfg)

        # Streamer
        cfg_new.setdefault("claude", {})
        cfg_new["claude"]["streamer_name"] = streamer_name.strip()
        cfg_new["claude"]["game_name"]     = game_name.strip()
        cfg_new["claude"]["content_type"]  = content_type

        # Módulos opcionales
        cfg_new.setdefault("face_detection", {})["enabled"] = face_enabled
        cfg_new.setdefault("subtitles", {})["enabled"]      = subtitles_enabled
        cfg_new.setdefault("layout", {})["detect_driving"]  = detect_driving
        cfg_new.setdefault("transcriber", {})["diarization_enabled"] = diarization_enabled

        # Whisper
        cfg_new.setdefault("whisper", {})
        cfg_new["whisper"]["model"]        = whisper_model
        cfg_new["whisper"]["language"]     = whisper_language.strip()
        cfg_new["whisper"]["compute_type"] = whisper_compute

        # Detección viral
        cfg_new.setdefault("viral_detection", {})
        cfg_new["viral_detection"]["min_clip_duration"]  = min_clip_dur
        cfg_new["viral_detection"]["max_clip_duration"]  = max_clip_dur
        cfg_new["viral_detection"]["pre_buffer_seconds"] = pre_buffer
        cfg_new["viral_detection"]["post_buffer_seconds"] = post_buffer
        cfg_new["viral_detection"]["top_n_clips"]        = top_n_clips
        cfg_new["viral_detection"]["skip_intro_sec"]     = skip_intro
        cfg_new["viral_detection"]["skip_outro_sec"]     = skip_outro
        cfg_new["viral_detection"]["viral_keywords"]     = [
            kw.strip() for kw in viral_keywords_raw.splitlines() if kw.strip()
        ]

        # Subtítulos
        cfg_new.setdefault("subtitles", {})
        cfg_new["subtitles"]["font_size"]        = sub_font_size
        cfg_new["subtitles"]["outline_width"]    = sub_outline
        cfg_new["subtitles"]["position_y_ratio"] = round(sub_pos_y, 3)
        cfg_new["subtitles"]["max_line_width"]   = sub_max_width
        cfg_new["subtitles"]["font_name"]        = sub_font_name.strip()
        cfg_new["subtitles"]["manual_offset_sec"] = round(sub_offset, 3)

        # Censura
        cfg_new.setdefault("censorship", {})
        cfg_new["censorship"]["mode"]              = cens_mode
        cfg_new["censorship"]["beep_frequency_hz"] = beep_freq
        cfg_new["censorship"]["custom_words"]      = [
            w.strip() for w in custom_words_raw.splitlines() if w.strip()
        ]

        # IA / Gemini
        cfg_new.setdefault("gemini", {})["model"]             = gemini_model.strip()
        cfg_new.setdefault("ai_features", {})["multimodal_video"]        = multimodal_video
        cfg_new["ai_features"]["multimodal_interval_sec"] = multimodal_interval

        # GPU
        cfg_new.setdefault("gpu", {})["force_cpu"] = force_cpu

        # Salida
        cfg_new.setdefault("output", {})
        cfg_new["output"]["fps"]             = output_fps
        cfg_new["output"]["orientation"]     = output_orientation
        cfg_new["output"]["naming_pattern"]  = naming_pattern.strip()

        _save_cfg(cfg_new)
        st.success("✅ Configuración guardada en `config.json`")

# ==========================================
# PESTAÑA 6: ANALÍTICA
# ==========================================
with tab_analytics:
    import altair as alt
    import pandas as pd
    import modules.analytics as _am
    from modules.analytics import (
        is_connected, connect_youtube, disconnect_youtube,
        get_channel_overview, get_analytics_with_delta,
        get_top_videos, get_day_of_week_stats,
    )

    _PERIODOS = ["1 día", "7 días", "14 días", "28 días", "1 mes", "2 meses", "3 meses", "4 meses", "5 meses", "6 meses", "Histórico"]
    _TIPOS    = ["Todos", "Shorts", "Vídeos"]

    # Paleta
    _CV = "#4C9BE8"; _CE = "#F59E0B"; _CS = "#2DD4BF"
    _CL = "#818CF8"; _CC = "#34D399"; _CSH = "#F472B6"; _CLO = "#F87171"

    def _delta_str(current, prev):
        if prev is None or prev == 0:
            return None
        pct = (current - prev) / abs(prev) * 100
        return f"{pct:+.1f}%"

    def _fmt_dur(seg):
        s = int(seg or 0)
        return f"{s // 60}m {s % 60:02d}s"

    # ── YouTube ────────────────────────────────────────────────────────────────
    st.subheader("📺 YouTube Analytics")

    from modules.analytics import list_accounts, set_active_account, get_active_account
    _accounts = list_accounts()
    if not _accounts:
        st.warning(
            "No se encontró ningún fichero de credenciales. "
            "Añade `client_secrets.json` o `client_secrets_NombreCuenta.json` "
            "descargado de Google Cloud Console → APIs & Services → Credentials "
            "→ OAuth 2.0 Client ID → tipo **Desktop app**."
        )
        st.stop()

    # ── Selector de cuenta ─────────────────────────────────────────────────
    _account_names = list(_accounts.keys())
    _cur_account   = get_active_account()
    _sel_account   = st.selectbox(
        "Cuenta YouTube",
        _account_names,
        index=_account_names.index(_cur_account) if _cur_account in _account_names else 0,
        key="yt_account",
    )
    if _sel_account != _cur_account:
        set_active_account(_sel_account)
        st.rerun()

    yt_connected = is_connected()

    if not yt_connected:
        st.info(
            "**Requisito previo (una sola vez):**  \n"
            "1. Credencial OAuth tipo **Desktop app** → `client_secrets.json` "
            "o `client_secrets_NombreCuenta.json`  \n"
            "2. *OAuth consent screen → Público → Usuarios de prueba* → añade tu email  \n"
            "3. Activa: **YouTube Data API v3** y **YouTube Analytics API**"
        )
        if st.button("🔗 Conectar YouTube", key="btn_connect_yt"):
            with st.spinner("Abriendo navegador para autenticación..."):
                ok = connect_youtube()
            if ok:
                st.success("✅ Conectado.")
                st.rerun()
            else:
                st.error("❌ No se pudo conectar. Revisa client_secrets.json.")
    else:
        # ── Header canal + filtros ─────────────────────────────────────────
        with st.spinner(""):
            overview = get_channel_overview()

        col_info, col_disc = st.columns([6, 1])
        with col_info:
            if overview:
                st.markdown(
                    f"**{overview['channelTitle']}** &nbsp;·&nbsp; "
                    f"**{overview['subscriberCount']:,}** suscriptores &nbsp;·&nbsp; "
                    f"**{overview['viewCount']:,}** vistas totales &nbsp;·&nbsp; "
                    f"**{overview['videoCount']:,}** vídeos"
                )
        with col_disc:
            if st.button("⛔ Desconectar", key="btn_disc_yt"):
                disconnect_youtube()
                st.rerun()

        col_tipo, col_periodo, _ = st.columns([2, 2, 4])
        with col_tipo:
            yt_tipo = st.selectbox("Tipo", _TIPOS, key="yt_tipo")
        with col_periodo:
            yt_periodo = st.selectbox("Periodo", _PERIODOS, index=2, key="yt_periodo")

        # Date pickers personalizados para Histórico
        if yt_periodo == "Histórico":
            today = datetime.date.today()
            default_start = datetime.date(2020, 1, 1)
            default_end   = (today.replace(day=1) - datetime.timedelta(days=1)).replace(day=1)
            col_ds, col_de = st.columns(2)
            with col_ds:
                hist_start = st.date_input(
                    "Desde (primer día del mes)",
                    value=default_start,
                    min_value=datetime.date(2005, 1, 1),
                    max_value=default_end,
                    key="hist_start",
                )
            with col_de:
                hist_end = st.date_input(
                    "Hasta (primer día del mes)",
                    value=default_end,
                    min_value=datetime.date(2005, 1, 1),
                    max_value=default_end,
                    key="hist_end",
                )
            # Selector de granularidad
            yt_granularity = st.radio(
                "Granularidad",
                ["Diario", "Mensual"],
                index=1,
                horizontal=True,
                key="hist_granularity",
            )
            yt_granularity = "day" if yt_granularity == "Diario" else "month"

            # Para mensual las fechas deben ser primer día de mes (requisito API)
            if yt_granularity == "month":
                hist_start = hist_start.replace(day=1)
                hist_end   = hist_end.replace(day=1)
            yt_periodo = f"custom:{hist_start}:{hist_end}"
        else:
            yt_granularity = "auto"

        st.divider()

        # ── Carga de datos ─────────────────────────────────────────────────
        with st.spinner("Cargando analytics..."):
            df, prev = get_analytics_with_delta(yt_periodo, yt_tipo, yt_granularity)
            df_top   = get_top_videos(yt_periodo, yt_tipo, limit=200)
            df_dow   = get_day_of_week_stats(yt_tipo)

        if yt_tipo != "Todos":
            st.caption(
                f"⚠ El filtro **{yt_tipo}** se aplica solo a la tabla de top vídeos. "
                "La YouTube Analytics API no permite filtrar series temporales por tipo de contenido."
            )

        if df is None:
            st.error("No se pudieron cargar los datos de analytics.")
            if _am.last_error:
                st.code(_am.last_error, language="text")
            st.stop()

        if df.empty:
            st.info("No hay datos en el periodo seleccionado.")
            st.stop()

        # ── KPIs con delta ─────────────────────────────────────────────────
        total_v   = int(df["vistas"].sum())
        total_min = int(df["minutos_visionados"].sum())
        total_sn  = int(df.get("suscriptores_ganados", pd.Series([0])).sum()
                        - df.get("suscriptores_perdidos", pd.Series([0])).sum())
        total_sh  = int(df.get("shares", pd.Series([0])).sum())
        avg_er    = float(df["engagement_rate"].mean()) if "engagement_rate" in df.columns else 0.0
        avg_dur   = float(df["duracion_media_seg"].mean()) if "duracion_media_seg" in df.columns else 0.0

        k1, k2, k3, k4, k5, k6 = st.columns(6)
        k1.metric("👁 Vistas",            f"{total_v:,}",
                  delta=_delta_str(total_v,   prev.get("vistas")   if prev else None))
        k2.metric("⏱ Horas visionadas",   f"{total_min // 60:,}h {total_min % 60}m",
                  delta=_delta_str(total_min, prev.get("minutos_visionados") if prev else None))
        k3.metric("💬 Eng. Rate",         f"{avg_er:.2f}%",
                  delta=_delta_str(avg_er,    prev.get("engagement_rate")    if prev else None))
        k4.metric("👥 Subs netos",        f"{total_sn:+,}",
                  delta=_delta_str(total_sn,
                      (prev.get("suscriptores_ganados", 0) - prev.get("suscriptores_perdidos", 0))
                      if prev else None))
        k5.metric("🔗 Shares",            f"{total_sh:,}",
                  delta=_delta_str(total_sh,  prev.get("shares")   if prev else None))
        k6.metric("▶ Dur. media",         _fmt_dur(avg_dur))

        if prev is None:
            st.caption("_(Sin comparativa disponible para 'Histórico')_")

        st.divider()

        # ── Gráficos de tendencia (4 tabs) ─────────────────────────────────
        t_vis, t_eng, t_subs, t_time = st.tabs(
            ["📈 Vistas", "💬 Engagement", "👥 Suscriptores", "⏱ Tiempo visionado"]
        )

        _x_type = "O" if yt_periodo != "Histórico" else "O"
        _x_title = "Fecha"

        with t_vis:
            ch = (
                alt.Chart(df).mark_area(opacity=0.25, color=_CV).encode(
                    x=alt.X("fecha:O", title=_x_title),
                    y=alt.Y("vistas:Q", title="Vistas"),
                ) + alt.Chart(df).mark_line(color=_CV, strokeWidth=2).encode(
                    x="fecha:O",
                    y="vistas:Q",
                    tooltip=["fecha", "vistas", "likes", "comentarios", "shares"],
                )
            ).properties(height=280)
            st.altair_chart(ch, width="stretch")

        with t_eng:
            df_eng = df[["fecha", "likes", "comentarios", "shares"]].melt(
                id_vars="fecha", var_name="tipo", value_name="valor"
            )
            _color_map = alt.Scale(
                domain=["likes", "comentarios", "shares"],
                range=[_CL, _CC, _CSH],
            )
            ch = (
                alt.Chart(df_eng)
                .mark_area(opacity=0.7)
                .encode(
                    x=alt.X("fecha:O", title=_x_title),
                    y=alt.Y("valor:Q", stack=True, title="Interacciones"),
                    color=alt.Color("tipo:N", scale=_color_map, title="Tipo"),
                    tooltip=["fecha", "tipo", "valor"],
                )
                .properties(height=280)
            )
            st.altair_chart(ch, width="stretch")
            st.caption(
                f"Engagement rate medio: **{avg_er:.2f}%** "
                f"({'sin comparativa' if prev is None else 'vs ' + str(round(prev.get('engagement_rate', 0), 2)) + '% periodo anterior'})"
            )

        with t_subs:
            df_subs = df[["fecha", "suscriptores_ganados", "suscriptores_perdidos"]].melt(
                id_vars="fecha", var_name="tipo", value_name="valor"
            )
            _subs_color = alt.Scale(
                domain=["suscriptores_ganados", "suscriptores_perdidos"],
                range=[_CS, _CLO],
            )
            ch = (
                alt.Chart(df_subs)
                .mark_bar(opacity=0.85)
                .encode(
                    x=alt.X("fecha:O", title=_x_title),
                    y=alt.Y("valor:Q", title="Suscriptores"),
                    color=alt.Color("tipo:N", scale=_subs_color, title=""),
                    tooltip=["fecha", "tipo", "valor"],
                )
                .properties(height=280)
            )
            st.altair_chart(ch, width="stretch")

        with t_time:
            df_t = df.copy()
            df_t["horas"] = df_t["minutos_visionados"] / 60
            ch = (
                alt.Chart(df_t).mark_area(opacity=0.25, color=_CE).encode(
                    x=alt.X("fecha:O", title=_x_title),
                    y=alt.Y("horas:Q", title="Horas visionadas"),
                ) + alt.Chart(df_t).mark_line(color=_CE, strokeWidth=2).encode(
                    x="fecha:O",
                    y="horas:Q",
                    tooltip=["fecha", "horas", "vistas"],
                )
            ).properties(height=280)
            st.altair_chart(ch, width="stretch")

        st.divider()

        # ── Mejor día para publicar ─────────────────────────────────────────
        st.markdown("### 📅 Mejor día para publicar _(últimos 90 días)_")
        if df_dow is not None and not df_dow.empty:
            col_a, col_b = st.columns(2)
            with col_a:
                ch_views = (
                    alt.Chart(df_dow)
                    .mark_bar(color=_CV, opacity=0.85)
                    .encode(
                        x=alt.X("vistas_media:Q", title="Media de vistas"),
                        y=alt.Y("dia_semana:N", sort=None, title=""),
                        tooltip=["dia_semana", "vistas_media", "n_dias"],
                    )
                    .properties(title="Media de vistas por día", height=220)
                )
                st.altair_chart(ch_views, width="stretch")
            with col_b:
                ch_er = (
                    alt.Chart(df_dow)
                    .mark_bar(color=_CE, opacity=0.85)
                    .encode(
                        x=alt.X("engagement_media:Q", title="Engagement Rate (%)"),
                        y=alt.Y("dia_semana:N", sort=None, title=""),
                        tooltip=["dia_semana", "engagement_media", "n_dias"],
                    )
                    .properties(title="Engagement Rate medio por día", height=220)
                )
                st.altair_chart(ch_er, width="stretch")

            # Recomendación automática
            best_views = df_dow.loc[df_dow["vistas_media"].idxmax(), "dia_semana"]
            best_er    = df_dow.loc[df_dow["engagement_media"].idxmax(), "dia_semana"]
            st.info(
                f"**Mejor día por vistas:** {best_views} &nbsp;|&nbsp; "
                f"**Mejor día por engagement:** {best_er}"
            )
        else:
            st.caption("Sin datos suficientes para el análisis por día de la semana.")

        st.divider()

        # ── Top vídeos ─────────────────────────────────────────────────────
        _n_top = len(df_top) if df_top is not None else 0
        st.markdown(f"### 🏆 Vídeos del periodo ({_n_top})")
        if df_top is not None and not df_top.empty:
            st.dataframe(
                df_top.style
                .format({
                    "vistas":       "{:,}",
                    "er_pct":       "{:.2f}%",
                    "likes":        "{:,}",
                    "comentarios":  "{:,}",
                    "shares":       "{:,}",
                    "subs_ganados": "{:,}",
                })
                .bar(subset=["vistas"], color=f"{_CV}33")
                .bar(subset=["er_pct"], color=f"{_CE}33"),
                width="stretch",
                hide_index=True,
                column_config={
                    "título":       st.column_config.TextColumn("Título", width="large"),
                    "vistas":       st.column_config.NumberColumn("Vistas"),
                    "er_pct":       st.column_config.NumberColumn("Tasa interacción %"),
                    "duracion_media":st.column_config.TextColumn("Duración media"),
                    "likes":        st.column_config.NumberColumn("Likes"),
                    "comentarios":  st.column_config.NumberColumn("Comentarios"),
                    "shares":       st.column_config.NumberColumn("Shares"),
                    "subs_ganados": st.column_config.NumberColumn("Subs ganados"),
                },
            )
        else:
            st.caption("Sin datos de vídeos para este periodo.")

        st.divider()

        # ── Carga de datos secundarios ─────────────────────────────────────
        from modules.analytics import (
            get_traffic_sources, get_subscribed_status,
            get_device_types, get_top_countries, get_demographics,
        )
        with st.spinner("Cargando análisis de audiencia..."):
            df_traffic  = get_traffic_sources(yt_periodo)
            df_subs_st  = get_subscribed_status(yt_periodo)
            df_devices  = get_device_types(yt_periodo)
            df_countries= get_top_countries(yt_periodo, limit=15)
            df_demo     = get_demographics(yt_periodo)

        # ── 1. Fuentes de tráfico ──────────────────────────────────────────
        st.markdown("### 🔎 Fuentes de tráfico")
        if df_traffic is not None and not df_traffic.empty:
            # Destacar fuente principal
            top_src = df_traffic.iloc[0]
            st.caption(
                f"Fuente principal: **{top_src['fuente']}** — "
                f"{top_src['vistas']:,} vistas ({top_src['porcentaje']}%)"
            )
            # Insight viral: Vídeos sugeridos
            sug = df_traffic[df_traffic["fuente"] == "Vídeos sugeridos"]
            if not sug.empty:
                pct_sug = sug.iloc[0]["porcentaje"]
                if pct_sug >= 40:
                    st.success(f"✅ El {pct_sug}% de tus vistas vienen de 'Vídeos sugeridos' — el algoritmo está distribuyendo tu contenido activamente.")
                elif pct_sug >= 20:
                    st.info(f"ℹ {pct_sug}% de vistas desde 'Vídeos sugeridos'. Aumentar el CTR y retención puede mejorar la distribución orgánica.")

            ch_traffic = (
                alt.Chart(df_traffic)
                .mark_bar(color=_CV, opacity=0.85)
                .encode(
                    x=alt.X("vistas:Q", title="Vistas"),
                    y=alt.Y("fuente:N", sort="-x", title=""),
                    tooltip=["fuente", "vistas", "porcentaje"],
                )
                .properties(height=max(200, len(df_traffic) * 32))
            )
            pct_text = (
                alt.Chart(df_traffic)
                .mark_text(align="left", dx=4, color="#8B8FA8")
                .encode(
                    x="vistas:Q",
                    y=alt.Y("fuente:N", sort="-x"),
                    text=alt.Text("porcentaje:Q", format=".1f", formatType="number"),
                )
            )
            st.altair_chart((ch_traffic + pct_text), width="stretch")
        else:
            st.caption("Sin datos de fuentes de tráfico.")

        st.divider()

        # ── 2. Suscriptores vs no suscriptores ─────────────────────────────
        st.markdown("### 🔔 Suscriptores vs no suscriptores")
        if df_subs_st is not None and not df_subs_st.empty:
            col_s1, col_s2 = st.columns([1, 2])
            with col_s1:
                for _, row in df_subs_st.iterrows():
                    st.metric(
                        row["estado"],
                        f"{row['vistas']:,} vistas",
                        delta=f"{row['porcentaje']}% del total",
                        delta_color="off",
                    )
                unsub = df_subs_st[df_subs_st["estado"] == "No suscriptores"]
                if not unsub.empty:
                    pct_ns = unsub.iloc[0]["porcentaje"]
                    if pct_ns >= 60:
                        st.success(f"✅ {pct_ns}% de vistas no suscriptores → alto alcance viral.")
                    elif pct_ns >= 40:
                        st.info(f"ℹ {pct_ns}% no suscriptores. Buen alcance externo.")
                    else:
                        st.warning(f"⚠ Solo {pct_ns}% no suscriptores. El contenido llega principalmente a tu audiencia actual.")
            with col_s2:
                ch_subs = (
                    alt.Chart(df_subs_st)
                    .mark_arc(innerRadius=55)
                    .encode(
                        theta=alt.Theta("vistas:Q"),
                        color=alt.Color("estado:N",
                            scale=alt.Scale(
                                domain=["Suscriptores", "No suscriptores"],
                                range=[_CS, _CV],
                            ),
                            legend=alt.Legend(title=""),
                        ),
                        tooltip=["estado", "vistas", "porcentaje"],
                    )
                    .properties(height=220)
                )
                st.altair_chart(ch_subs, width="stretch")

            # Tabla comparativa de métricas
            df_subs_display = df_subs_st.copy()
            df_subs_display["horas_visionadas"] = (
                df_subs_display["minutos_visionados"] // 60
            ).astype(str) + "h " + (
                df_subs_display["minutos_visionados"] % 60
            ).astype(str) + "m"
            df_subs_display["duración_media"] = df_subs_display["duracion_media_seg"].apply(
                lambda s: f"{int(s)//60}m {int(s)%60:02d}s"
            )
            st.dataframe(
                df_subs_display[["estado", "vistas", "porcentaje", "horas_visionadas", "duración_media"]]
                .style.format({"vistas": "{:,}", "porcentaje": "{:.1f}%"}),
                hide_index=True, width="stretch",
            )
        else:
            st.caption("Sin datos de estado de suscripción.")

        st.divider()

        # ── 3. Tipos de dispositivo ────────────────────────────────────────
        st.markdown("### 📱 Dispositivos")
        if df_devices is not None and not df_devices.empty:
            col_d1, col_d2 = st.columns([1, 1])
            with col_d1:
                dev_tab_v, dev_tab_h = st.tabs(["Vistas", "Tiempo de visualización"])
                with dev_tab_v:
                    ch_dev = (
                        alt.Chart(df_devices)
                        .mark_arc(innerRadius=50)
                        .encode(
                            theta=alt.Theta("vistas:Q"),
                            color=alt.Color("dispositivo:N",
                                scale=alt.Scale(scheme="tableau10"),
                                legend=alt.Legend(title=""),
                            ),
                            tooltip=["dispositivo", "vistas", "porcentaje", "horas_visionadas"],
                        )
                        .properties(height=240)
                    )
                    st.altair_chart(ch_dev, width="stretch")
                with dev_tab_h:
                    ch_dev_h = (
                        alt.Chart(df_devices)
                        .mark_arc(innerRadius=50)
                        .encode(
                            theta=alt.Theta("horas_visionadas:Q"),
                            color=alt.Color("dispositivo:N",
                                scale=alt.Scale(scheme="tableau10"),
                                legend=alt.Legend(title=""),
                            ),
                            tooltip=["dispositivo", "horas_visionadas", "vistas"],
                        )
                        .properties(height=240)
                    )
                    st.altair_chart(ch_dev_h, width="stretch")
            with col_d2:
                st.dataframe(
                    df_devices.style.format({
                        "vistas":           "{:,}",
                        "porcentaje":       "{:.1f}%",
                        "horas_visionadas": "{:.1f}h",
                    }),
                    hide_index=True, width="stretch",
                    column_config={
                        "dispositivo":      st.column_config.TextColumn("Dispositivo"),
                        "vistas":           st.column_config.NumberColumn("Vistas"),
                        "porcentaje":       st.column_config.NumberColumn("% Vistas"),
                        "horas_visionadas": st.column_config.NumberColumn("Horas"),
                    },
                )
                mobile = df_devices[df_devices["dispositivo"] == "Móvil"]
                if not mobile.empty and mobile.iloc[0]["porcentaje"] >= 70:
                    st.success(f"✅ {mobile.iloc[0]['porcentaje']}% de vistas en móvil — el formato vertical 9:16 es la decisión correcta.")
        else:
            st.caption("Sin datos de dispositivos.")

        st.divider()

        # ── 4. Distribución geográfica ─────────────────────────────────────
        st.markdown("### 🌍 Distribución geográfica")
        if df_countries is not None and not df_countries.empty:
            df_countries = df_countries.copy()
            df_countries["horas_visionadas"] = (df_countries["minutos_visionados"] / 60).round(1)
            col_g1, col_g2 = st.columns([2, 1])
            with col_g1:
                geo_tab_v, geo_tab_h = st.tabs(["Vistas", "Tiempo de visualización"])
                with geo_tab_v:
                    ch_geo = (
                        alt.Chart(df_countries.head(10))
                        .mark_bar(color=_CS, opacity=0.85)
                        .encode(
                            x=alt.X("vistas:Q", title="Vistas"),
                            y=alt.Y("país:N", sort="-x", title=""),
                            tooltip=["país", "vistas", "porcentaje", "horas_visionadas"],
                        )
                        .properties(height=300)
                    )
                    pct_geo = (
                        alt.Chart(df_countries.head(10))
                        .mark_text(align="left", dx=4, color="#8B8FA8")
                        .encode(
                            x="vistas:Q",
                            y=alt.Y("país:N", sort="-x"),
                            text=alt.Text("porcentaje:Q", format=".1f"),
                        )
                    )
                    st.altair_chart(ch_geo + pct_geo, width="stretch")
                with geo_tab_h:
                    ch_geo_h = (
                        alt.Chart(df_countries.head(10))
                        .mark_bar(color=_CE, opacity=0.85)
                        .encode(
                            x=alt.X("horas_visionadas:Q", title="Horas visionadas"),
                            y=alt.Y("país:N", sort="-x", title=""),
                            tooltip=["país", "horas_visionadas", "vistas"],
                        )
                        .properties(height=300)
                    )
                    h_text = (
                        alt.Chart(df_countries.head(10))
                        .mark_text(align="left", dx=4, color="#8B8FA8")
                        .encode(
                            x="horas_visionadas:Q",
                            y=alt.Y("país:N", sort="-x"),
                            text=alt.Text("horas_visionadas:Q", format=".1f"),
                        )
                    )
                    st.altair_chart(ch_geo_h + h_text, width="stretch")
            with col_g2:
                st.dataframe(
                    df_countries[["país", "vistas", "porcentaje", "horas_visionadas"]]
                    .style.format({"vistas": "{:,}", "porcentaje": "{:.1f}%", "horas_visionadas": "{:.1f}h"}),
                    hide_index=True, width="stretch",
                    column_config={
                        "país":             st.column_config.TextColumn("País"),
                        "vistas":           st.column_config.NumberColumn("Vistas"),
                        "porcentaje":       st.column_config.NumberColumn("% Vistas"),
                        "horas_visionadas": st.column_config.NumberColumn("Horas"),
                    },
                )
        else:
            st.caption("Sin datos geográficos.")

        st.divider()

        # ── 5. Demografía de audiencia ─────────────────────────────────────
        st.markdown("### 👥 Demografía de audiencia")
        st.caption("_(Sin filtro por tipo de contenido — limitación de la API de YouTube)_")
        if df_demo is not None and not df_demo.empty:
            _gender_colors = alt.Scale(
                domain=["Hombre", "Mujer", "No especificado"],
                range=[_CV, _CSH, "#94A3B8"],
            )
            ch_demo = (
                alt.Chart(df_demo)
                .mark_bar(opacity=0.85)
                .encode(
                    x=alt.X("edad:N",
                        sort=["13 – 17", "18 – 24", "25 – 34",
                              "35 – 44", "45 – 54", "55 – 64", "65+"],
                        title="Grupo de edad",
                    ),
                    y=alt.Y("porcentaje:Q", title="% de audiencia"),
                    color=alt.Color("genero:N", scale=_gender_colors, title="Género"),
                    xOffset="genero:N",
                    tooltip=["edad", "genero", "porcentaje"],
                )
                .properties(height=300)
            )
            st.altair_chart(ch_demo, width="stretch")

            # Resumen
            top_seg = df_demo.loc[df_demo["porcentaje"].idxmax()]
            st.caption(
                f"Segmento dominante: **{top_seg['edad']} años, {top_seg['genero']}** "
                f"({top_seg['porcentaje']:.1f}%)"
            )
        else:
            st.caption("Sin datos demográficos disponibles.")

        st.divider()

    # ── TikTok ─────────────────────────────────────────────────────────────────
    st.subheader("🎵 TikTok Analytics")
    st.info(
        "La TikTok for Business API requiere aprobación manual "
        "(solicitud en developers.tiktok.com). "
        "Una vez aprobada, añade el Access Token en **⚙️ Configuración** → TikTok."
    )
    tiktok_token = config.get("tiktok", {}).get("access_token", "").strip()
    if tiktok_token:
        st.success("Token detectado — integración en desarrollo.")
