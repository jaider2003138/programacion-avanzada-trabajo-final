"""
Componentes reutilizables para la interfaz Streamlit.
"""

from __future__ import annotations

import base64
import math
from collections import Counter
from html import escape
from pathlib import Path
from textwrap import dedent
from typing import Any, Callable

import pandas as pd
import streamlit as st


ASSETS_DIR = Path(__file__).resolve().parent / "assets"
LOGO_PATH = ASSETS_DIR / "logo.svg"

try:
    from app.api.prediction_utils import (
        HIGH_CONFIDENCE_LEVEL,
        LOW_CONFIDENCE_LEVEL,
        MEDIUM_CONFIDENCE_LEVEL,
        calculate_confidence_metrics,
        get_confidence_level,
        normalize_confidence_percent,
    )
except (ImportError, ModuleNotFoundError):
    HIGH_CONFIDENCE_LEVEL = "Alta confianza"
    MEDIUM_CONFIDENCE_LEVEL = "Confianza media"
    LOW_CONFIDENCE_LEVEL = "Baja confianza"

    def normalize_confidence_percent(confidence: float | int | str) -> float:
        confidence_value = float(confidence)
        if confidence_value <= 1:
            confidence_value *= 100
        return min(max(confidence_value, 0.0), 100.0)

    def get_confidence_level(confidence_percent: float) -> str:
        normalized_confidence = normalize_confidence_percent(confidence_percent)
        if normalized_confidence >= 80:
            return HIGH_CONFIDENCE_LEVEL
        if normalized_confidence >= 50:
            return MEDIUM_CONFIDENCE_LEVEL
        return LOW_CONFIDENCE_LEVEL

    def calculate_confidence_metrics(confidences: list[float]) -> dict[str, float | int]:
        normalized_confidences: list[float] = []
        for confidence in confidences:
            try:
                normalized_confidence = normalize_confidence_percent(confidence)
            except (TypeError, ValueError):
                continue

            if math.isfinite(normalized_confidence):
                normalized_confidences.append(normalized_confidence)

        total = len(normalized_confidences)
        average_confidence = (
            sum(normalized_confidences) / total
            if total
            else 0.0
        )

        return {
            "total": total,
            "average_confidence": average_confidence,
            "high_confidence": sum(
                1 for confidence in normalized_confidences if confidence >= 80
            ),
            "medium_confidence": sum(
                1 for confidence in normalized_confidences
                if 50 <= confidence < 80
            ),
            "low_confidence": sum(
                1 for confidence in normalized_confidences if confidence < 50
            ),
        }


def _clean_markup(markup: str) -> str:
    """
    Quita sangria para que Markdown no trate el HTML como bloque de codigo.
    """
    lines = dedent(markup).strip().splitlines()
    return " ".join(line.strip() for line in lines if line.strip())


def _render_html(markup: str) -> None:
    """
    Renderiza HTML limpio dentro de Streamlit.
    """
    st.markdown(_clean_markup(markup), unsafe_allow_html=True)


def _logo_markup() -> str:
    """
    Devuelve el logo embebido como data URI si el asset existe.
    """
    if not LOGO_PATH.exists():
        return '<div class="logo-mark">▣</div>'

    encoded = base64.b64encode(LOGO_PATH.read_bytes()).decode("ascii")
    return (
        '<img class="logo-image" '
        f'src="data:image/svg+xml;base64,{encoded}" '
        'alt="Logo del sistema" />'
    )


def render_app_header() -> None:
    """
    Muestra el encabezado principal de la aplicacion.
    """
    logo_markup = _logo_markup()

    st.markdown(
        f"""
        <section class="app-hero">
            <div class="app-brand">
                {logo_markup}
                <div>
                    <h1 class="app-title">Sistema Inteligente de Inventario</h1>
                    <p class="app-subtitle">
                        Clasificacion e inventariado de productos de despensa mediante imagenes
                    </p>
                </div>
            </div>
            <div class="header-actions">
                <span class="header-pill">ONNX Runtime</span>
                <span class="header-pill">API Flask</span>
                <span class="header-pill">Streamlit</span>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_status_banner(is_online: bool) -> None:
    """
    Muestra el estado de conexion con la API.
    """
    if is_online:
        css_class = "success"
        icon = "✓"
        message = "API Flask conectada correctamente."
    else:
        css_class = "error"
        icon = "!"
        message = "La API Flask no esta activa. Ejecuta primero: python -m app.api.app"

    st.markdown(
        f"""
        <div class="status-banner {css_class}">
            <span class="status-icon">{icon}</span>
            <span>{escape(message)}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_panel_title(title: str, icon: str = "") -> None:
    """
    Renderiza un titulo de panel consistente.
    """
    icon_html = f"<span>{escape(icon)}</span>" if icon else ""
    st.markdown(
        f'<div class="panel-title">{icon_html}<span>{escape(title)}</span></div>',
        unsafe_allow_html=True,
    )


def render_small_panel_title(title: str, icon: str = "") -> None:
    """
    Renderiza un titulo pequeno de panel.
    """
    icon_html = f"<span>{escape(icon)}</span>" if icon else ""
    st.markdown(
        f'<div class="panel-title small">{icon_html}<span>{escape(title)}</span></div>',
        unsafe_allow_html=True,
    )


def render_field_label(text: str) -> None:
    """
    Renderiza una etiqueta visual para grupos de controles.
    """
    st.markdown(f'<div class="panel-kicker">{escape(text)}</div>', unsafe_allow_html=True)


def render_empty_preview() -> None:
    """
    Placeholder para la vista previa del producto.
    """
    st.markdown(
        """
        <div class="preview-placeholder">
            <div class="preview-icon">▧</div>
            <div>Sin imagen</div>
        </div>
        <div class="preview-caption">Vista previa del producto</div>
        """,
        unsafe_allow_html=True,
    )


def _stat_card(label: str, value: str, badge: str, color: str) -> str:
    return _clean_markup(
        f"""
        <div class="result-stat">
            <div>
                <div class="result-label">{escape(label)}</div>
                <div class="result-value">{escape(value)}</div>
            </div>
            <span class="result-badge {color}">{escape(badge)}</span>
        </div>
        """
    )


def _confidence_badge_class(level: str) -> str:
    if level == HIGH_CONFIDENCE_LEVEL:
        return "confidence-high"
    if level == MEDIUM_CONFIDENCE_LEVEL:
        return "confidence-medium"
    return "confidence-low"


def _confidence_stat_badge_class(level: str) -> str:
    if level == HIGH_CONFIDENCE_LEVEL:
        return "badge-green"
    if level == MEDIUM_CONFIDENCE_LEVEL:
        return "badge-yellow"
    return "badge-red"


def render_confidence_level_badge(level: str) -> str:
    """
    Devuelve una etiqueta HTML para el nivel de confianza.
    """
    css_class = _confidence_badge_class(level)
    return (
        f'<span class="confidence-level-badge {css_class}">'
        f'{escape(level)}</span>'
    )


def _format_confidence_percent(value: Any) -> str:
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return "0.00%"

    if not math.isfinite(confidence):
        return "0.00%"

    return f"{confidence:.2f}%"


def _confidence_metric_card(label: str, value: str, css_class: str) -> str:
    return _clean_markup(
        f"""
        <div class="confidence-metric-card {css_class}">
            <div>
                <div class="confidence-metric-label">{escape(label)}</div>
                <div class="confidence-metric-value">{escape(value)}</div>
            </div>
            <span class="confidence-metric-icon"></span>
        </div>
        """
    )


def render_confidence_metric_cards(
    metrics: dict[str, Any],
    *,
    include_total: bool = True,
    include_average: bool = True,
) -> None:
    """
    Renderiza tarjetas con metricas agregadas de confianza.
    """
    items: list[tuple[str, str, str]] = []

    if include_total:
        items.append(("Total procesados", str(metrics.get("total", 0)), "confidence-total"))
    if include_average:
        items.append(
            (
                "Confianza promedio",
                _format_confidence_percent(metrics.get("average_confidence", 0)),
                "confidence-average",
            )
        )

    items.extend(
        [
            ("Alta confianza", str(metrics.get("high_confidence", 0)), "confidence-high"),
            ("Confianza media", str(metrics.get("medium_confidence", 0)), "confidence-medium"),
            ("Baja confianza", str(metrics.get("low_confidence", 0)), "confidence-low"),
        ]
    )

    columns = st.columns(len(items))
    for column, (label, value, css_class) in zip(columns, items):
        with column:
            _render_html(_confidence_metric_card(label, value, css_class))


def get_confidence_message(confidence_percent: float | None) -> str:
    """
    Devuelve una recomendacion breve segun la confianza del modelo.
    """
    if confidence_percent is None:
        return "-"

    level = get_confidence_level(confidence_percent)
    if level == HIGH_CONFIDENCE_LEVEL:
        return "Predicción confiable"
    if level == MEDIUM_CONFIDENCE_LEVEL:
        return "Revisar antes de guardar"
    return "Corregir manualmente"


def render_result_cards(
    prediction_response: dict[str, Any] | None,
    format_category: Callable[[str], str],
) -> None:
    """
    Renderiza las tarjetas de resultado del panel de analisis.
    """
    if prediction_response is None:
        category = "-"
        confidence = "-%"
        confidence_level = "-"
        confidence_badge_class = "badge-blue"
    else:
        prediction = prediction_response["prediction"]
        category = format_category(prediction["predicted_category"])
        confidence_value = prediction.get("confidence_percent")
        if confidence_value in (None, ""):
            confidence_value = prediction.get("confidence", 0)
        confidence_percent = normalize_confidence_percent(confidence_value)
        confidence = f"{confidence_percent:.2f}%"
        confidence_level = (
            prediction.get("confidence_level")
            or get_confidence_level(confidence_percent)
        )
        confidence_badge_class = _confidence_stat_badge_class(confidence_level)

    _render_html(
        f"""
        <div class="result-stack">
            {_stat_card("Categoria predicha", category, "◇", "badge-red")}
            {_stat_card("Confianza", confidence, "⌁", "badge-blue")}
            {_stat_card("Nivel de confianza", confidence_level, "!", confidence_badge_class)}
        </div>
        """,
    )


def render_result_details(
    prediction_response: dict[str, Any] | None,
    format_category: Callable[[str], str],
) -> None:
    """
    Muestra el detalle de top predicciones cuando existe un resultado.
    """
    with st.expander(
        "Ver detalles del analisis",
        expanded=prediction_response is not None,
    ):
        if prediction_response is None:
            st.caption("Clasifica una imagen para ver el detalle del modelo.")
            return

        prediction = prediction_response["prediction"]
        generated_code = prediction_response["generated_code"]
        st.write(f"Codigo generado: `{generated_code}`")

        top_data = [
            {
                "Categoria": format_category(item["category"]),
                "Confianza (%)": round(
                    normalize_confidence_percent(
                        item.get("confidence_percent")
                        if item.get("confidence_percent") not in (None, "")
                        else item.get("confidence", 0)
                    ),
                    2,
                ),
            }
            for item in prediction.get("top_predictions", [])[:3]
        ]
        st.dataframe(pd.DataFrame(top_data), use_container_width=True)


def format_file_size(size_bytes: int | None) -> str:
    """
    Convierte bytes a una etiqueta corta.
    """
    if not size_bytes:
        return "0 KB"
    if size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    return f"{size_bytes / (1024 * 1024):.1f} MB"


def _file_thumb_markup(file_obj: Any, fallback_icon: str) -> str:
    """
    Construye una miniatura embebida para imagenes cargadas.
    """
    file_type = getattr(file_obj, "type", "") or ""
    if not file_type.startswith("image/"):
        return f'<span class="batch-file-icon">{escape(fallback_icon)}</span>'

    try:
        encoded = base64.b64encode(file_obj.getvalue()).decode("ascii")
    except Exception:
        return f'<span class="batch-file-icon">{escape(fallback_icon)}</span>'

    return (
        '<img class="batch-file-thumb" '
        f'src="data:{escape(file_type)};base64,{encoded}" '
        f'alt="{escape(getattr(file_obj, "name", "archivo"))}" />'
    )


def _batch_status_for_filename(
    filename: str,
    batch_result: dict[str, Any] | None,
) -> tuple[str, str]:
    """
    Devuelve el estado visual de un archivo segun el resultado real de la API.
    """
    if not batch_result:
        return "uploaded", "Cargada"

    for result in batch_result.get("results", []):
        if result.get("filename") == filename:
            if "error" in result:
                return "error", "Error"
            return "processed", "Procesada"

    return "uploaded", "Cargada"


def _batch_status_for_zip(batch_result: dict[str, Any] | None) -> tuple[str, str]:
    """
    Devuelve el estado visual del ZIP cargado.
    """
    if not batch_result:
        return "uploaded", "Cargado"

    total_received = int(batch_result.get("total_received", 0))
    total_skipped = int(batch_result.get("total_skipped", 0))
    if total_received > 0 and total_skipped == total_received:
        return "error", "Error"
    return "processed", "Procesado"


def render_batch_file_list(
    files: list[Any] | None = None,
    zip_file: Any | None = None,
    batch_result: dict[str, Any] | None = None,
) -> None:
    """
    Renderiza una lista compacta de archivos para el cargue masivo.
    """
    rows: list[str] = []

    if files:
        for file_obj in files:
            status_class, status_text = _batch_status_for_filename(
                file_obj.name,
                batch_result,
            )
            rows.append(
                _clean_markup(
                    f"""
                <div class="batch-file-row">
                    <div class="batch-file-main">
                        {_file_thumb_markup(file_obj, "▧")}
                        <span class="batch-file-name">{escape(file_obj.name)}</span>
                    </div>
                    <span class="batch-file-size">{format_file_size(getattr(file_obj, "size", 0))}</span>
                    <span class="batch-status {status_class}">{status_text}</span>
                    <span class="batch-remove">×</span>
                </div>
                """
                )
            )

    elif zip_file is not None:
        status_class, status_text = _batch_status_for_zip(batch_result)
        rows.append(
            _clean_markup(
                f"""
            <div class="batch-file-row">
                <div class="batch-file-main">
                    <span class="batch-file-icon">▤</span>
                    <span class="batch-file-name">{escape(zip_file.name)}</span>
                </div>
                <span class="batch-file-size">{format_file_size(getattr(zip_file, "size", 0))}</span>
                <span class="batch-status {status_class}">{status_text}</span>
                <span class="batch-remove">×</span>
            </div>
            """
            )
        )

    if not rows:
        st.caption("Aun no hay archivos seleccionados.")
        return

    _render_html(
        f'<div class="batch-file-list">{"".join(rows)}</div>',
    )


def _summary_row(label: str, value: str, icon: str, color: str) -> str:
    return _clean_markup(
        f"""
        <div class="batch-summary-row">
            <div class="batch-summary-label">
                <span class="batch-summary-icon {color}">{escape(icon)}</span>
                <span>{escape(label)}</span>
            </div>
            <strong>{escape(value)}</strong>
        </div>
        """
    )


def _batch_counts(
    files: list[Any] | None,
    zip_file: Any | None,
    batch_result: dict[str, Any] | None,
) -> tuple[int, int, bool, str, int]:
    loaded_count = len(files or []) + (1 if zip_file is not None else 0)
    zip_detected = zip_file is not None

    if batch_result:
        valid_count = int(batch_result.get("total_processed", 0))
        total_received = int(batch_result.get("total_received", loaded_count))
        skipped = int(batch_result.get("total_skipped", 0))
        status = "Procesado" if skipped == 0 else "Con alertas"
        progress = 100 if total_received else 0
        return total_received, valid_count, zip_detected, status, progress

    status = "Listo para procesar" if loaded_count else "Esperando archivos"
    valid_count = len(files or [])
    return loaded_count, valid_count, zip_detected, status, 0


def render_batch_summary(
    files: list[Any] | None,
    zip_file: Any | None,
    batch_result: dict[str, Any] | None,
    format_category: Callable[[str], str],
    expected_categories: list[str] | None = None,
) -> None:
    """
    Renderiza el panel lateral de resumen para cargue masivo.
    """
    loaded_count, valid_count, zip_detected, status, progress = _batch_counts(
        files,
        zip_file,
        batch_result,
    )
    status_class = "ok" if status in {"Listo para procesar", "Procesado"} else "muted"
    category_counter: Counter[str] = Counter()

    if batch_result:
        for result in batch_result.get("results", []):
            if "error" not in result:
                category_counter[format_category(result["predicted_category"])] += 1

    chips = ""
    if category_counter:
        category_title = "Categorias detectadas"
        chips = "".join(
            f'<span class="category-chip">{escape(category)} ({count})</span>'
            for category, count in category_counter.most_common()
        )
    else:
        category_title = "Categorias esperadas"
        categories = expected_categories or [
            "Arroz y granos",
            "Pastas",
            "Aceites",
            "Salsas y condimentos",
            "Cafe y chocolate",
            "Enlatados",
            "Azucar y sal",
        ]
        chips = "".join(
            f'<span class="category-chip muted">{escape(category)}</span>'
            for category in categories
        )

    _render_html(
        f"""
        <div class="batch-summary">
            {_summary_row("Archivos cargados", str(loaded_count), "▧", "blue")}
            {_summary_row("Imagenes validas", str(valid_count), "▧", "green")}
            {_summary_row("ZIP detectado", "Si" if zip_detected else "No", "▤", "orange")}
            <div class="batch-summary-row">
                <div class="batch-summary-label">
                    <span class="batch-summary-icon purple">↻</span>
                    <span>Estado</span>
                </div>
                <strong class="batch-state {status_class}">{escape(status)}</strong>
            </div>
            <div class="batch-progress-block">
                <div class="batch-progress-head">
                    <span>Progreso del procesamiento</span>
                    <strong>{progress}%</strong>
                </div>
                <div class="batch-progress-track">
                    <span style="width: {progress}%"></span>
                </div>
                <div class="batch-progress-caption">
                    {"Completado" if progress else "Esperando inicio"}
                </div>
            </div>
            <div class="batch-category-block">
                <div class="batch-category-title">{category_title}</div>
                <div class="batch-category-chips">{chips}</div>
            </div>
        </div>
        """,
    )


def render_batch_details(
    batch_result: dict[str, Any] | None,
    format_category: Callable[[str], str],
) -> None:
    """
    Muestra detalle del lote procesado.
    """
    with st.expander("Ver detalles del lote"):
        if not batch_result:
            st.caption("Procesa un lote para ver el detalle de resultados.")
            return

        rows = []
        for result in batch_result.get("results", []):
            if "error" in result:
                rows.append(
                    {
                        "Archivo": result["filename"],
                        "Categoria": "Error",
                        "Confianza (%)": "-",
                        "Nivel de confianza": "-",
                        "Codigo": "-",
                        "Estado": result["error"],
                    }
                )
            else:
                confidence_value = result.get("confidence_percent")
                if confidence_value in (None, ""):
                    confidence_value = result.get("confidence", 0)
                confidence_percent = normalize_confidence_percent(confidence_value)
                confidence_level = (
                    result.get("confidence_level")
                    or get_confidence_level(confidence_percent)
                )
                rows.append(
                    {
                        "Archivo": result["filename"],
                        "Categoria": format_category(result["predicted_category"]),
                        "Confianza (%)": round(confidence_percent, 2),
                        "Nivel de confianza": confidence_level,
                        "Codigo": result["generated_code"],
                        "Estado": "Procesado",
                    }
                )

        st.dataframe(pd.DataFrame(rows), use_container_width=True)


def render_info_strip(title: str, copy: str) -> None:
    """
    Muestra una franja informativa inferior.
    """
    st.markdown(
        f"""
        <div class="info-strip">
            <span class="info-icon">i</span>
            <div>
                <div class="info-title">{escape(title)}</div>
                <div class="info-copy">{escape(copy)}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
