"""
analytics.py
Integración con YouTube Analytics API v2 y YouTube Data API v3.
Dashboard profesional de analítica viral para la pestaña Analítica de la UI.
"""

import logging
import pickle
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

_SCOPES = [
    "https://www.googleapis.com/auth/youtube.readonly",
    "https://www.googleapis.com/auth/yt-analytics.readonly",
]
_active_account: str = ""

# Último error de API expuesto a la UI
last_error: str = ""


def list_accounts() -> dict:
    """Devuelve {nombre: Path} para cada client_secrets*.json encontrado."""
    accounts: dict = {}
    legacy = Path("client_secrets.json")
    if legacy.exists():
        accounts["default"] = legacy
    for p in sorted(Path(".").glob("client_secrets_*.json")):
        name = p.stem[len("client_secrets_"):]
        accounts[name] = p
    return accounts


def set_active_account(name: str) -> None:
    global _active_account
    _active_account = name


def get_active_account() -> str:
    return _active_account


# Inicializar al primer account disponible al importar
_accs_init = list_accounts()
if _accs_init:
    _active_account = next(iter(_accs_init))

_METRICS_DAILY = [
    "views",
    "estimatedMinutesWatched",
    "averageViewDuration",
    "likes",
    "subscribersGained",
    "subscribersLost",
    "shares",
    "comments",
]

_RENAME = {
    "day":                       "fecha",
    "month":                     "fecha",
    "views":                     "vistas",
    "estimatedMinutesWatched":   "minutos_visionados",
    "averageViewDuration":       "duracion_media_seg",
    "likes":                     "likes",
    "subscribersGained":         "suscriptores_ganados",
    "subscribersLost":           "suscriptores_perdidos",
    "shares":                    "shares",
    "comments":                  "comentarios",
}


# ── Autenticación ──────────────────────────────────────────────────────────────

def _resolve_account_files() -> tuple:
    """Devuelve (secrets_file, token_file) para el account activo."""
    accounts = list_accounts()
    if not accounts:
        return None, None
    name = _active_account if _active_account in accounts else next(iter(accounts))
    secrets_file = accounts[name]
    token_file   = Path(f"token_analytics_{name}.pickle")
    return secrets_file, token_file


def _get_yt_credentials():
    try:
        from google.auth.transport.requests import Request
        from google_auth_oauthlib.flow import InstalledAppFlow
    except ImportError:
        logger.error("Ejecuta: pip install google-api-python-client google-auth-oauthlib")
        return None

    secrets_file, token_file = _resolve_account_files()
    if secrets_file is None:
        return None

    credentials = None
    if token_file.exists():
        with open(token_file, "rb") as f:
            credentials = pickle.load(f)

    if credentials and credentials.valid:
        return credentials

    if credentials and credentials.expired and credentials.refresh_token:
        try:
            credentials.refresh(Request())
            with open(token_file, "wb") as f:
                pickle.dump(credentials, f)
            return credentials
        except Exception as exc:
            logger.warning(f"No se pudo refrescar token: {exc}")
            token_file.unlink(missing_ok=True)

    flow = InstalledAppFlow.from_client_secrets_file(str(secrets_file), _SCOPES)
    credentials = flow.run_local_server(port=0, prompt="consent", open_browser=True)
    with open(token_file, "wb") as f:
        pickle.dump(credentials, f)
    return credentials


def is_connected() -> bool:
    _, token_file = _resolve_account_files()
    if token_file is None or not token_file.exists():
        return False
    try:
        with open(token_file, "rb") as f:
            creds = pickle.load(f)
        return creds is not None and (creds.valid or bool(creds.refresh_token))
    except Exception:
        return False


def connect_youtube() -> bool:
    try:
        return _get_yt_credentials() is not None
    except Exception as exc:
        logger.error(f"Error conectando: {exc}")
        return False


def disconnect_youtube():
    _, token_file = _resolve_account_files()
    if token_file is not None:
        token_file.unlink(missing_ok=True)


# ── Rangos de fecha ────────────────────────────────────────────────────────────

def date_range(period: str) -> tuple[str, str]:
    """
    Devuelve (start, end) en formato YYYY-MM-DD.
    Acepta también el formato especial "custom:YYYY-MM-DD:YYYY-MM-DD"
    para rangos personalizados (usado en Histórico con date pickers).
    Para dimensions=month, ambas fechas deben ser primer día de mes.
    """
    if period.startswith("custom:"):
        _, s, e = period.split(":", 2)
        return s, e

    today            = date.today()
    yesterday        = today - timedelta(days=1)
    last_month_start = (today.replace(day=1) - timedelta(days=1)).replace(day=1)

    def _months_ago(n: int) -> date:
        """Primer día del mes de hace N meses."""
        m = today.month - n
        y = today.year + m // 12
        m = m % 12 or 12
        if m == 12:
            y -= 1
        return date(y, m, 1)

    mapping = {
        "1 día":    (yesterday,             yesterday),
        "7 días":   (today - timedelta(7),  yesterday),
        "14 días":  (today - timedelta(14), yesterday),
        "28 días":  (today - timedelta(28), yesterday),
        "1 mes":    (today.replace(day=1),  yesterday),
        "2 meses":  (_months_ago(2),        yesterday),
        "3 meses":  (_months_ago(3),        yesterday),
        "4 meses":  (_months_ago(4),        yesterday),
        "5 meses":  (_months_ago(5),        yesterday),
        "6 meses":  (_months_ago(6),        yesterday),
        "Histórico":(date(2020, 1, 1),      last_month_start),
    }
    s, e = mapping.get(period, (today - timedelta(7), yesterday))
    return str(s), str(e)


def _prev_date_range(period: str) -> Optional[tuple[str, str]]:
    """Devuelve el rango equivalente del periodo ANTERIOR para calcular deltas."""
    today     = date.today()
    yesterday = today - timedelta(days=1)

    def _first_of_month(d: date, offset: int) -> date:
        """Primer día del mes desplazado offset meses desde d."""
        m = d.month + offset
        y = d.year + (m - 1) // 12
        m = (m - 1) % 12 + 1
        return date(y, m, 1)

    if period == "1 día":
        d = yesterday - timedelta(1)
        return str(d), str(d)
    if period == "7 días":
        return str(today - timedelta(14)), str(today - timedelta(8))
    if period == "14 días":
        return str(today - timedelta(28)), str(today - timedelta(15))
    if period == "28 días":
        return str(today - timedelta(56)), str(today - timedelta(29))
    if period == "1 mes":
        first_this = today.replace(day=1)
        last_prev  = first_this - timedelta(1)
        return str(last_prev.replace(day=1)), str(last_prev)

    # N meses: periodo anterior = mismo número de meses N meses antes
    _mes_map = {"2 meses": 2, "3 meses": 3, "4 meses": 4, "5 meses": 5, "6 meses": 6}
    if period in _mes_map:
        n = _mes_map[period]
        start_curr = _first_of_month(today, -n)
        start_prev = _first_of_month(today, -2 * n)
        end_prev   = start_curr - timedelta(1)
        return str(start_prev), str(end_prev)

    return None  # Histórico y custom


# ── Core query ─────────────────────────────────────────────────────────────────

def _query_report(
    creds,
    start: str,
    end: str,
    dimensions: str,
    metrics: list[str],
    video_type: str = "Todos",
    sort: str = None,
    max_results: int = None,
) -> Optional[pd.DataFrame]:
    """Wrapper central sobre youtubeAnalytics v2 reports.query().

    NOTA: el filtro videoType solo es válido con dimensions='video'.
    Para day/month se ignora (limitación de la API de YouTube Analytics).
    """
    global last_error
    # videoType filter solo funciona con dimensions="video"
    vt_filter = None
    if dimensions == "video":
        vt_filter = {"Shorts": "videoType==shortVideoType",
                     "Vídeos": "videoType==regularVideoType"}.get(video_type)
    try:
        from googleapiclient.discovery import build
        svc = build("youtubeAnalytics", "v2", credentials=creds)
        params: dict = dict(
            ids="channel==MINE",
            startDate=start,
            endDate=end,
            dimensions=dimensions,
            metrics=",".join(metrics),
        )
        if sort:        params["sort"]       = sort
        if max_results: params["maxResults"] = max_results
        if vt_filter:   params["filters"]    = vt_filter

        resp    = svc.reports().query(**params).execute()
        headers = [h["name"] for h in resp.get("columnHeaders", [])]
        rows    = resp.get("rows", [])
        if not rows:
            return pd.DataFrame(columns=headers)

        df = pd.DataFrame(rows, columns=headers)
        # Las columnas de dimensión son strings; solo convertir las de métricas
        dim_cols = set(dimensions.split(",")) | {"day", "month", "video"}
        for col in df.columns:
            if col not in dim_cols:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
        return df

    except Exception as exc:
        last_error = str(exc)
        logger.error(f"YouTube Analytics query error: {exc}")
        return None


# ── YouTube Data API v3 ────────────────────────────────────────────────────────

def get_channel_overview() -> Optional[dict]:
    creds = _get_yt_credentials()
    if not creds:
        return None
    try:
        from googleapiclient.discovery import build
        yt   = build("youtube", "v3", credentials=creds)
        resp = yt.channels().list(part="snippet,statistics", mine=True).execute()
        items = resp.get("items", [])
        if not items:
            return None
        item  = items[0]
        stats = item.get("statistics", {})
        return {
            "channelId":      item["id"],
            "channelTitle":   item["snippet"]["title"],
            "subscriberCount":int(stats.get("subscriberCount", 0)),
            "viewCount":      int(stats.get("viewCount", 0)),
            "videoCount":     int(stats.get("videoCount", 0)),
        }
    except Exception as exc:
        logger.error(f"Error overview canal: {exc}")
        return None


def _get_video_titles(creds, video_ids: list[str]) -> dict[str, str]:
    if not video_ids:
        return {}
    try:
        from googleapiclient.discovery import build
        yt     = build("youtube", "v3", credentials=creds)
        titles = {}
        for i in range(0, len(video_ids), 50):
            chunk = video_ids[i:i+50]
            resp  = yt.videos().list(part="snippet", id=",".join(chunk)).execute()
            for item in resp.get("items", []):
                titles[item["id"]] = item["snippet"]["title"]
        return titles
    except Exception as exc:
        logger.warning(f"No se pudieron obtener títulos: {exc}")
        return {}


# ── Analytics públicas ─────────────────────────────────────────────────────────

def get_analytics_with_delta(
    period: str, video_type: str = "Todos", granularity: str = "auto"
) -> tuple[Optional[pd.DataFrame], Optional[dict]]:
    """
    Retorna (df_current, prev_totals).
    df_current: datos del periodo con columnas en español + engagement_rate.
    prev_totals: dict {metrica: valor} del periodo anterior, o None.

    granularity: "auto" (día para periodos cortos, mes para Histórico),
                 "day" (fuerza diario), "month" (fuerza mensual).
    Nota: si granularity="month" las fechas deben ser primer día de mes.
    """
    creds = _get_yt_credentials()
    if not creds:
        return None, None

    is_hist = period in ("Histórico",) or period.startswith("custom:")
    if granularity == "auto":
        dim = "month" if is_hist else "day"
    else:
        dim = granularity
    start, end = date_range(period)

    df = _query_report(creds, start, end, dim, _METRICS_DAILY,
                       video_type, sort=dim)
    if df is None:
        return None, None

    # Renombrar
    df = df.rename(columns={k: v for k, v in _RENAME.items() if k in df.columns})

    # Engagement rate por fila
    if "vistas" in df.columns and df["vistas"].sum() > 0:
        df["engagement_rate"] = (
            df.get("likes", 0) + df.get("comentarios", 0) + df.get("shares", 0)
        ) / df["vistas"].replace(0, float("nan")) * 100
        df["engagement_rate"] = df["engagement_rate"].fillna(0).round(2)
    else:
        df["engagement_rate"] = 0.0

    # Periodo anterior
    prev_totals = None
    prev_range  = _prev_date_range(period)
    if prev_range:
        df_prev = _query_report(creds, prev_range[0], prev_range[1],
                                "day", _METRICS_DAILY, video_type)
        if df_prev is not None and not df_prev.empty:
            prev_totals = {
                "vistas":               int(df_prev.get("views",                   pd.Series([0])).sum()),
                "minutos_visionados":   int(df_prev.get("estimatedMinutesWatched", pd.Series([0])).sum()),
                "likes":                int(df_prev.get("likes",                   pd.Series([0])).sum()),
                "suscriptores_ganados": int(df_prev.get("subscribersGained",       pd.Series([0])).sum()),
                "suscriptores_perdidos":int(df_prev.get("subscribersLost",         pd.Series([0])).sum()),
                "shares":               int(df_prev.get("shares",                  pd.Series([0])).sum()),
                "comentarios":          int(df_prev.get("comments",                pd.Series([0])).sum()),
            }
            prev_v = prev_totals["vistas"]
            prev_e = (prev_totals["likes"] + prev_totals["comentarios"] + prev_totals["shares"])
            prev_totals["engagement_rate"] = (prev_e / prev_v * 100) if prev_v else 0.0

    return df, prev_totals


def get_day_of_week_stats(video_type: str = "Todos") -> Optional[pd.DataFrame]:
    """
    Últimos 90 días agrupados por día de la semana.
    Retorna DataFrame: {dia_semana, dia_num, vistas_media, engagement_media, n_dias}
    """
    creds = _get_yt_credentials()
    if not creds:
        return None

    today    = date.today()
    start    = str(today - timedelta(days=90))
    end      = str(today - timedelta(days=1))

    df = _query_report(creds, start, end, "day", _METRICS_DAILY, video_type, sort="day")
    if df is None or df.empty:
        return None

    df["_fecha"] = pd.to_datetime(df["day"])
    df["dia_num"] = df["_fecha"].dt.dayofweek  # 0=Lun … 6=Dom
    df["_engagement"] = (
        df.get("likes", 0) + df.get("comments", 0) + df.get("shares", 0)
    ) / df["views"].replace(0, float("nan")) * 100

    _DIAS = {0: "Lun", 1: "Mar", 2: "Mié", 3: "Jue", 4: "Vie", 5: "Sáb", 6: "Dom"}

    agg = (
        df.groupby("dia_num")
        .agg(
            vistas_media    =("views",        "mean"),
            engagement_media=("_engagement",  "mean"),
            n_dias          =("views",        "count"),
        )
        .reset_index()
    )
    agg["dia_semana"] = agg["dia_num"].map(_DIAS)
    agg["vistas_media"]     = agg["vistas_media"].round(0).astype(int)
    agg["engagement_media"] = agg["engagement_media"].fillna(0).round(2)
    return agg.sort_values("dia_num")


def get_top_videos(
    period: str, video_type: str = "Todos", limit: int = 200
) -> Optional[pd.DataFrame]:
    """
    Top N vídeos por vistas con métricas completas y engagement rate calculado.
    Columnas: título, vistas, er_pct, duracion_media, likes, comentarios, shares, subs_ganados
    """
    creds = _get_yt_credentials()
    if not creds:
        return None

    start, end = date_range(period)
    metrics    = [
        "views", "likes", "comments", "shares",
        "estimatedMinutesWatched", "averageViewDuration", "subscribersGained",
    ]

    df = _query_report(creds, start, end, "video", metrics, video_type,
                       sort="-views", max_results=limit)
    if df is None or df.empty:
        return pd.DataFrame()

    # Títulos reales
    titles     = _get_video_titles(creds, df["video"].tolist())
    df["título"] = df["video"].map(lambda v: titles.get(v, v))

    # Engagement rate
    df["er_pct"] = (
        (df.get("likes", 0) + df.get("comments", 0) + df.get("shares", 0))
        / df["views"].replace(0, float("nan")) * 100
    ).fillna(0).round(2)

    # Duración media legible
    def _fmt_dur(seg):
        s = int(seg or 0)
        return f"{s // 60}m {s % 60:02d}s"
    df["duracion_media"] = df["averageViewDuration"].apply(_fmt_dur)

    int_cols = ["views", "likes", "comments", "shares",
                "estimatedMinutesWatched", "subscribersGained"]
    for c in int_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)

    return df[[
        "título", "views", "er_pct", "duracion_media",
        "likes", "comments", "shares", "subscribersGained",
    ]].rename(columns={
        "views":             "vistas",
        "comments":          "comentarios",
        "subscribersGained": "subs_ganados",
    })


# ── Fuentes de tráfico ─────────────────────────────────────────────────────────

_TRAFFIC_LABELS = {
    "YT_SEARCH":          "Búsqueda en YouTube",
    "SUGGESTED_VIDEOS":   "Vídeos sugeridos",
    "EXT_URL":            "URL externa",
    "NO_LINK_EMBEDDED":   "Embebido externo",
    "NO_LINK_OTHER":      "Directo / sin enlace",
    "NOTIFICATION":       "Notificaciones",
    "PLAYLIST":           "Playlists",
    "YT_CHANNEL":         "Página del canal",
    "YT_OTHER_PAGE":      "Otra página de YouTube",
    "SUBSCRIBER":         "Feed de suscripciones",
    "END_SCREEN":         "Pantalla final",
    "YT_ADVERTISING":     "Publicidad",
    "SHORTS_FEED":        "Feed de Shorts",
    "HASHTAGS":           "Hashtags",
    "PRODUCT_PAGE":       "Página de producto",
}


def get_traffic_sources(period: str) -> Optional[pd.DataFrame]:
    """
    Fuentes de tráfico del canal en el periodo dado.
    Retorna DataFrame: {fuente, vistas, pct}
    """
    creds = _get_yt_credentials()
    if not creds:
        return None
    start, end = date_range(period)
    df = _query_report(creds, start, end, "insightTrafficSourceType",
                       ["views"], sort=None)
    if df is None or df.empty:
        return df
    df = df.rename(columns={"insightTrafficSourceType": "fuente_raw", "views": "vistas"})
    df["fuente"] = df["fuente_raw"].map(lambda x: _TRAFFIC_LABELS.get(x, x))
    df["vistas"] = df["vistas"].astype(int)
    total = df["vistas"].sum()
    df["porcentaje"] = ((df["vistas"] / total * 100).round(1) if total > 0 else 0.0).fillna(0)
    return df[["fuente", "vistas", "porcentaje"]].sort_values("vistas", ascending=False)


# ── Suscriptores vs no suscriptores ───────────────────────────────────────────

def get_subscribed_status(period: str) -> Optional[pd.DataFrame]:
    """
    Comparativa suscriptores / no suscriptores.
    Retorna DataFrame: {estado, vistas, minutos_visionados, duracion_media_seg}
    """
    creds = _get_yt_credentials()
    if not creds:
        return None
    start, end = date_range(period)
    # subscribedStatus no soporta videoType filter
    df = _query_report(creds, start, end, "subscribedStatus",
                       ["views", "estimatedMinutesWatched", "averageViewDuration"])
    if df is None or df.empty:
        return df
    df = df.rename(columns={
        "subscribedStatus":      "estado_raw",
        "views":                 "vistas",
        "estimatedMinutesWatched": "minutos_visionados",
        "averageViewDuration":   "duracion_media_seg",
    })
    df["estado"] = df["estado_raw"].map({
        "SUBSCRIBED":   "Suscriptores",
        "UNSUBSCRIBED": "No suscriptores",
    }).fillna(df["estado_raw"])
    for col in ["vistas", "minutos_visionados"]:
        df[col] = df[col].astype(int)
    total = df["vistas"].sum()
    df["porcentaje"] = ((df["vistas"] / total * 100).round(1) if total > 0 else 0.0).fillna(0)
    return df[["estado", "vistas", "porcentaje", "minutos_visionados", "duracion_media_seg"]]


# ── Tipos de dispositivo ───────────────────────────────────────────────────────

_DEVICE_LABELS = {
    "MOBILE":           "Móvil",
    "DESKTOP":          "PC / Escritorio",
    "TABLET":           "Tablet",
    "TV":               "Televisión",
    "GAME_CONSOLE":     "Consola",
    "UNKNOWN_PLATFORM": "Desconocido",
}


def get_device_types(period: str) -> Optional[pd.DataFrame]:
    """
    Distribución de vistas por tipo de dispositivo.
    Retorna DataFrame: {dispositivo, vistas, porcentaje, horas_visionadas}
    """
    creds = _get_yt_credentials()
    if not creds:
        return None
    start, end = date_range(period)
    df = _query_report(creds, start, end, "deviceType", ["views", "estimatedMinutesWatched"])
    if df is None or df.empty:
        return df
    df = df.rename(columns={
        "deviceType":              "dispositivo_raw",
        "views":                   "vistas",
        "estimatedMinutesWatched": "minutos_visionados",
    })
    df["dispositivo"] = df["dispositivo_raw"].map(
        lambda x: _DEVICE_LABELS.get(x, x)
    )
    df["vistas"] = df["vistas"].astype(int)
    df["minutos_visionados"] = df["minutos_visionados"].astype(int)
    df["horas_visionadas"] = (df["minutos_visionados"] / 60).round(1)
    total = df["vistas"].sum()
    df["porcentaje"] = ((df["vistas"] / total * 100).round(1) if total > 0 else 0.0).fillna(0)
    return df[["dispositivo", "vistas", "porcentaje", "horas_visionadas"]].sort_values("vistas", ascending=False)


# ── Distribución geográfica ────────────────────────────────────────────────────

_COUNTRY_NAMES = {
    "ES": "España", "MX": "México", "AR": "Argentina", "CO": "Colombia",
    "CL": "Chile",  "PE": "Perú",   "VE": "Venezuela", "US": "Estados Unidos",
    "BR": "Brasil", "EC": "Ecuador","BO": "Bolivia",   "PY": "Paraguay",
    "UY": "Uruguay","GT": "Guatemala","HN": "Honduras","SV": "El Salvador",
    "NI": "Nicaragua","CR": "Costa Rica","PA": "Panamá","DO": "Rep. Dominicana",
    "CU": "Cuba",   "PR": "Puerto Rico","GB": "Reino Unido","DE": "Alemania",
    "FR": "Francia","IT": "Italia", "PT": "Portugal", "NL": "Países Bajos",
    "CA": "Canadá", "AU": "Australia","JP": "Japón",  "KR": "Corea del Sur",
    "IN": "India",  "PH": "Filipinas","RU": "Rusia",  "PL": "Polonia",
}


def get_top_countries(period: str, limit: int = 15) -> Optional[pd.DataFrame]:
    """
    Países con más vistas en el periodo.
    Retorna DataFrame: {pais, codigo, vistas, minutos_visionados, pct}
    """
    creds = _get_yt_credentials()
    if not creds:
        return None
    start, end = date_range(period)
    df = _query_report(creds, start, end, "country",
                       ["views", "estimatedMinutesWatched"],
                       sort="-views", max_results=limit)
    if df is None or df.empty:
        return df
    df = df.rename(columns={
        "country":               "codigo",
        "views":                 "vistas",
        "estimatedMinutesWatched": "minutos_visionados",
    })
    df["país"]    = df["codigo"].map(lambda x: _COUNTRY_NAMES.get(x, x))
    df["vistas"]  = df["vistas"].astype(int)
    df["minutos_visionados"] = df["minutos_visionados"].astype(int)
    total = df["vistas"].sum()
    df["porcentaje"] = ((df["vistas"] / total * 100).round(1) if total > 0 else 0.0).fillna(0)
    return df[["país", "codigo", "vistas", "porcentaje", "minutos_visionados"]].sort_values(
        "vistas", ascending=False
    )


# ── Demografía de audiencia ────────────────────────────────────────────────────

def get_demographics(period: str) -> Optional[pd.DataFrame]:
    """
    Distribución de audiencia por grupo de edad y género.
    Retorna DataFrame: {edad, genero, porcentaje}
    Nota: viewerPercentage no soporta videoType filter.
    """
    creds = _get_yt_credentials()
    if not creds:
        return None
    start, end = date_range(period)
    # ageGroup+gender no soportan videoType filter ni sort
    df = _query_report(creds, start, end, "ageGroup,gender",
                       ["viewerPercentage"])
    if df is None or df.empty:
        return df
    df = df.rename(columns={"viewerPercentage": "porcentaje"})
    df["edad"] = df["ageGroup"].str.replace("age", "").str.replace("-", " – ").str.replace(
        r"(\d+)$", r"\1+", regex=True
    )
    df["genero"] = df["gender"].map({
        "male":           "Hombre",
        "female":         "Mujer",
        "user_specified": "No especificado",
    }).fillna(df["gender"])
    df["porcentaje"] = pd.to_numeric(df["porcentaje"], errors="coerce").fillna(0).round(2)
    return df[["edad", "genero", "porcentaje"]].sort_values(
        ["edad", "genero"]
    )
