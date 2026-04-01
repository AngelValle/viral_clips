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

tab_lanzador, tab_revisor, tab_cache, tab_tools = st.tabs([
    "🚀 1. Lanzador de Tareas",
    "✂️ 2. Revisión y Publicación",
    "🗑️ 3. Caché",
    "🔧 4. Herramientas",
])

# ── Estado global ─────────────────────────────────────────────────────────────
if "global_log" not in st.session_state:
    st.session_state["global_log"] = []
if "running" not in st.session_state:
    st.session_state["running"] = False

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

def _run_with_log(cmd: list, log_placeholder):
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True, encoding="utf-8", errors="replace", env=env) as proc:
        for line in proc.stdout:
            _append_log(line.rstrip())
            log_placeholder.markdown(_render_log_md(st.session_state["global_log"]), unsafe_allow_html=True)
    return proc.returncode

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
                                     disabled=not parts_to_delete, use_container_width=True,
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
                                     type="secondary", use_container_width=True):
                            _shutil.rmtree(vc, ignore_errors=True)
                            _append_log(f"💣 Caché eliminada: {vc.name}")
                            st.success(f"✅ Caché de {vc.name} eliminada.")
                            st.rerun()

        st.divider()
        if st.button("💣 Limpiar TODO", type="secondary", use_container_width=True):
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
    if st.button("▶ Ejecutar diagnóstico GPU", key="btn_gpu", use_container_width=True):
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

    if st.button("▶ Abrir selector de webcam", key="btn_webcam", use_container_width=True):
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

    if st.button("▶ Medir border ratio", key="btn_border", use_container_width=True):
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


# ── Log único — definido aquí para estar disponible en ambas pestañas ──────────
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
st.divider()

# ==========================================
# PESTAÑA 1: LANZADOR MULTI-VÍDEO
# ==========================================
with tab_lanzador:
    st.markdown("### Selecciona qué hacer con cada vídeo de la carpeta:")

    opciones_pasos = {
        1: "1. Solo Transcribir (Whisper)",
        2: "2. Detección Viral (Recomendado para revisar luego)",
        3: "3. Extraer Segmentos Crudos",
        5: "5. Composición 9:16 Visual",
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
    elif st.button("▶️ Lanzar Procesamiento Seleccionado", type="primary", use_container_width=True):
        if not tareas_a_ejecutar:
            st.error("No has marcado ningún vídeo para procesar.")
        else:
            st.session_state["running"] = True
            for video_file, max_step in tareas_a_ejecutar.items():
                _append_log(f"▶ Iniciando: {video_file.name} (paso {max_step})")
                cmd = ["python", "-u", "main.py", "--file", str(video_file.resolve()), "--max-step", str(max_step)]
                rc = _run_with_log(cmd, LOG_PLACEHOLDER)
                if rc != 0:
                    _append_log(f"❌ Falló: {video_file.name} (código {rc})")
                    st.error(f"❌ Error en {video_file.name} (código {rc})")
                else:
                    _append_log(f"✅ Completado: {video_file.name}")
                    st.success(f"✅ {video_file.name} completado.")
            st.session_state["running"] = False
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

            # ── Columna derecha: edición y publicación (sin form aquí) ────────
            with col_edit:
                frags = seg.get("fragments", [])
                is_multi = frags and len(frags) > 1

                st.markdown("#### ⚙️ Edición")

                if is_multi:
                    st.caption(f"{len(frags)} fragmentos — duración total: {sum(f['end']-f['start'] for f in frags):.1f}s")
                    for fi, frag in enumerate(frags):
                        st.markdown(f"**Fragmento {fi+1}:** `{frag['start']:.1f}s → {frag['end']:.1f}s`")
                else:
                    total_dur = float(seg["end"]) - float(seg["start"])
                    st.caption(f"⏱ {seg['start']:.1f}s — {seg['end']:.1f}s ({total_dur:.1f}s)")

                st.markdown("#### 🚀 Publicación")
                st.caption("Marca las plataformas en el formulario de abajo.")

            st.divider()

        # ── Formulario: solo recoge aprobación, tiempos y publicación ─────────
        with st.form("review_form"):
            updated_segments = []

            for i, seg in enumerate(segments):
                frags = seg.get("fragments", [])
                is_multi = frags and len(frags) > 1

                st.markdown(f"**Clip {i+1}** — {seg.get('gemini_desc', '')}")
                aprobado = st.checkbox("✅ Aprobar", value=True, key=f"chk_{i}")

                if is_multi:
                    # Slider por fragmento
                    new_frags = []
                    for fi, frag in enumerate(frags):
                        f_min = max(0.0, float(frag["start"]) - 10.0)
                        f_max = float(frag["end"]) + 10.0
                        fs, fe = st.slider(
                            f"Fragmento {fi+1}:",
                            min_value=f_min, max_value=f_max,
                            value=(float(frag["start"]), float(frag["end"])),
                            step=0.1, key=f"frag_{i}_{fi}"
                        )
                        new_frags.append({"start": fs, "end": fe})
                    new_start = new_frags[0]["start"]
                    new_end = new_frags[-1]["end"]
                else:
                    new_start, new_end = st.slider(
                        "Tiempos (s):",
                        min_value=max(0.0, float(seg["start"]) - 30.0),
                        max_value=float(seg["end"]) + 30.0,
                        value=(float(seg["start"]), float(seg["end"])),
                        step=0.1, key=f"slider_{i}"
                    )
                    new_frags = None

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
                    if new_frags:
                        updated_seg["fragments"] = new_frags
                    elif updated_seg.get("fragments"):
                        updated_seg["fragments"][0]["start"] = new_start
                        updated_seg["fragments"][-1]["end"] = new_end
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
            elif st.button("🎬 Renderizar y Publicar Clips Aprobados (Pasos 3 al 8)", type="primary", use_container_width=True):
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
