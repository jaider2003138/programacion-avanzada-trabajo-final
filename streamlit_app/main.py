"""
Interfaz Streamlit para el Sistema Inteligente de Clasificación
e Inventariado de Productos de Despensa.

Para ejecutar:

Terminal 1:
    python -m app.api.app

Terminal 2:
    streamlit run streamlit_app/main.py
"""

from __future__ import annotations

import io
import json
import os
from datetime import datetime
from html import escape
from pathlib import Path
from typing import Any

import pandas as pd
import requests
import streamlit as st


try:
    from streamlit_app.auth.login import show_login_page
    from streamlit_app.auth.register import show_register_page
except ModuleNotFoundError:
    from auth.login import show_login_page
    from auth.register import show_register_page

try:
    from streamlit_app.components import (
        calculate_confidence_metrics,
        get_confidence_level,
        normalize_confidence_percent,
        render_app_header,
        render_batch_details,
        render_batch_file_list,
        render_batch_summary,
        render_confidence_level_badge,
        render_confidence_metric_cards,
        render_empty_preview,
        render_field_label,
        render_info_strip,
        render_panel_title,
        render_result_cards,
        render_result_details,
        render_small_panel_title,
        render_status_banner,
    )
    from streamlit_app.styles import apply_custom_styles
except ModuleNotFoundError:
    from components import (
        calculate_confidence_metrics,
        get_confidence_level,
        normalize_confidence_percent,
        render_app_header,
        render_batch_details,
        render_batch_file_list,
        render_batch_summary,
        render_confidence_level_badge,
        render_confidence_metric_cards,
        render_empty_preview,
        render_field_label,
        render_info_strip,
        render_panel_title,
        render_result_cards,
        render_result_details,
        render_small_panel_title,
        render_status_banner,
    )
    from styles import apply_custom_styles


API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:5000")


CATEGORY_LABELS = {
    "arroz_y_granos": "Arroz y granos",
    "pastas": "Pastas",
    "aceites": "Aceites",
    "salsas_y_condimentos": "Salsas y condimentos",
    "cafe_chocolate": "Café y chocolate",
    "enlatados": "Enlatados",
    "azucar_sal": "Azúcar y sal",
}

DEFAULT_USER_NAME = "Usuario local"
DEFAULT_USER_EMAIL = "local@app.com"
DEFAULT_USER_ROLE = "bodega"

AUDIT_ACTION_LABELS = {
    "producto_clasificado": "Producto clasificado",
    "producto_agregado": "Producto agregado",
    "producto_actualizado": "Producto actualizado",
    "cargue_masivo_realizado": "Cargue masivo realizado",
    "inventario_consultado": "Inventario consultado",
    "producto_desactivado": "Producto desactivado",
    "usuario_creado": "Usuario creado",
    "usuario_actualizado": "Usuario actualizado",
}


def is_logged_in() -> bool:
    """
    Indica si hay un usuario autenticado en la sesion.
    """
    return bool(st.session_state.get("user_id"))


def current_user_role() -> str:
    """
    Devuelve el rol del usuario autenticado.
    """
    return str(st.session_state.get("user_role") or DEFAULT_USER_ROLE)


def get_current_user() -> dict[str, str]:
    """
    Devuelve el usuario actual para enviarlo a la API.
    """
    user_name = str(st.session_state.get("user_name") or "").strip()
    user_email = str(st.session_state.get("user_email") or "").strip()
    user_role = str(st.session_state.get("user_role") or "").strip()

    return {
        "user_id": str(st.session_state.get("user_id") or ""),
        "user_name": user_name or DEFAULT_USER_NAME,
        "user_email": user_email or DEFAULT_USER_EMAIL,
        "user_role": user_role or DEFAULT_USER_ROLE,
    }


def set_logged_user(user: dict[str, Any]) -> None:
    """
    Guarda el usuario autenticado en session_state.
    """
    st.session_state["user_id"] = user["id"]
    st.session_state["user_name"] = user["full_name"]
    st.session_state["user_email"] = user["email"]
    st.session_state["user_role"] = user["role"]


def clear_logged_user() -> None:
    """
    Cierra la sesion local.
    """
    for key in ("user_id", "user_name", "user_email", "user_role"):
        st.session_state.pop(key, None)


def render_authenticated_sidebar() -> None:
    """
    Muestra informacion de sesion y cierre.
    """
    with st.sidebar:
        st.markdown("### Sesion")
        st.write(f"**{st.session_state.get('user_name', DEFAULT_USER_NAME)}**")
        st.caption(st.session_state.get("user_email", DEFAULT_USER_EMAIL))
        st.markdown(f"`{current_user_role()}`")
        if st.button("Cerrar sesion", use_container_width=True):
            clear_logged_user()
            st.rerun()


def format_audit_action(action: str | None) -> str:
    """
    Convierte la accion tecnica a una etiqueta legible.
    """
    if not action:
        return "-"
    return AUDIT_ACTION_LABELS.get(action, action.replace("_", " ").capitalize())


def check_api_status() -> bool:
    """
    Verifica si la API Flask está activa.
    """
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200
    except requests.RequestException:
        return False


def predict_image(uploaded_file) -> dict[str, Any] | None:
    """
    Envía una imagen al endpoint /predict.
    """
    try:
        files = {
            "image": (
                uploaded_file.name,
                uploaded_file.getvalue(),
                uploaded_file.type,
            )
        }

        response = requests.post(
            f"{API_BASE_URL}/predict",
            files=files,
            data=get_current_user(),
            timeout=60,
        )

        if response.status_code != 200:
            st.error("Error al clasificar la imagen.")
            st.json(response.json())
            return None

        return response.json()

    except requests.RequestException as exc:
        st.error(f"No se pudo conectar con la API: {exc}")
        return None


def register_product(
    payload: dict[str, Any],
    *,
    show_errors: bool = True,
) -> dict[str, Any] | None:
    """
    Envía un producto al endpoint /products.
    """
    try:
        request_payload = {
            **payload,
            **get_current_user(),
        }
        response = requests.post(
            f"{API_BASE_URL}/products",
            json=request_payload,
            timeout=30,
        )

        if response.status_code not in [200, 201]:
            if show_errors:
                st.error("Error al registrar el producto.")
                try:
                    st.json(response.json())
                except Exception:
                    st.write(response.text)
            return None

        return response.json()

    except requests.RequestException as exc:
        if show_errors:
            st.error(f"No se pudo registrar el producto: {exc}")
        return None


def get_products(*, audit: bool = False) -> list[dict[str, Any]]:
    """
    Consulta productos registrados.
    """
    try:
        params = get_current_user()
        if audit:
            params["audit"] = "true"

        response = requests.get(
            f"{API_BASE_URL}/products",
            params=params,
            timeout=10,
        )

        if response.status_code != 200:
            return []

        data = response.json()
        return data.get("products", [])

    except requests.RequestException:
        return []


def get_audit_logs(filters: dict[str, str]) -> list[dict[str, Any]]:
    """
    Consulta los registros de auditoria desde la API.

    Reglas:
    - Admin: puede ver todos los logs y aplicar filtros.
    - Bodega: la API limita automaticamente la consulta a sus propios logs.
    """
    try:
        params = {
            key: value
            for key, value in filters.items()
            if value
        }
        params["limit"] = "200"

        current_user = get_current_user()

        headers = {
            "X-User-Id": current_user["user_id"],
            "X-User-Name": current_user["user_name"],
            "X-User-Email": current_user["user_email"],
            "X-User-Role": current_user["user_role"],
        }

        response = requests.get(
            f"{API_BASE_URL}/audit/logs",
            params=params,
            headers=headers,
            timeout=15,
        )

        if response.status_code != 200:
            st.warning("No se pudieron consultar los logs de auditoria.")
            try:
                st.json(response.json())
            except Exception:
                st.write(response.text)
            return []

        return response.json().get("logs", [])

    except requests.RequestException as exc:
        st.warning(f"No se pudo conectar con la auditoria: {exc}")
        return []


def get_users() -> list[dict[str, Any]]:
    """
    Consulta usuarios registrados. Solo admin.
    """
    try:
        response = requests.get(
            f"{API_BASE_URL}/users",
            params=get_current_user(),
            timeout=15,
        )

        if response.status_code != 200:
            st.warning("No se pudieron consultar los usuarios.")
            return []

        return response.json().get("users", [])

    except requests.RequestException as exc:
        st.warning(f"No se pudo conectar con usuarios: {exc}")
        return []


def create_app_user(payload: dict[str, Any]) -> dict[str, Any] | None:
    """
    Crea usuario desde la API. Solo admin.
    """
    try:
        response = requests.post(
            f"{API_BASE_URL}/users",
            json={**payload, **get_current_user()},
            timeout=20,
        )

        if response.status_code not in (200, 201):
            try:
                st.error(response.json().get("detail") or response.json().get("error"))
            except Exception:
                st.error("No se pudo crear el usuario.")
            return None

        return response.json().get("user")

    except requests.RequestException as exc:
        st.error(f"No se pudo crear el usuario: {exc}")
        return None


def update_product_api(code: str, payload: dict[str, Any]) -> dict[str, Any] | None:
    """
    Actualiza un producto. Solo admin.
    """
    try:
        response = requests.patch(
            f"{API_BASE_URL}/products/{code}",
            json={**payload, **get_current_user()},
            timeout=20,
        )

        if response.status_code != 200:
            try:
                st.error(response.json().get("detail") or response.json().get("error"))
            except Exception:
                st.error("No se pudo actualizar el producto.")
            return None

        return response.json().get("product")

    except requests.RequestException as exc:
        st.error(f"No se pudo actualizar el producto: {exc}")
        return None


def deactivate_product_api(code: str) -> dict[str, Any] | None:
    """
    Desactiva un producto. Solo admin.
    """
    try:
        response = requests.delete(
            f"{API_BASE_URL}/products/{code}",
            json=get_current_user(),
            timeout=20,
        )

        if response.status_code != 200:
            try:
                st.error(response.json().get("detail") or response.json().get("error"))
            except Exception:
                st.error("No se pudo desactivar el producto.")
            return None

        return response.json().get("product")

    except requests.RequestException as exc:
        st.error(f"No se pudo desactivar el producto: {exc}")
        return None


def format_category(category: str) -> str:
    """
    Convierte el nombre técnico de la categoría a un nombre legible.
    """
    return CATEGORY_LABELS.get(category, category)


def confidence_percent_from_result(result: dict[str, Any]) -> float:
    """
    Obtiene la confianza de un resultado en formato porcentaje.
    """
    confidence_value = result.get("confidence_percent")
    if confidence_value in (None, ""):
        confidence_value = result.get("confidence", 0)

    return normalize_confidence_for_display(confidence_value)


def normalize_confidence_for_display(value: Any) -> float:
    """
    Normaliza una confianza y protege la UI contra valores vacios o NaN.
    """
    try:
        confidence_percent = normalize_confidence_percent(value)
    except (TypeError, ValueError):
        return 0.0

    if pd.isna(confidence_percent):
        return 0.0

    return float(confidence_percent)


def confidence_level_from_result(result: dict[str, Any]) -> str:
    """
    Obtiene el nivel de confianza de un resultado de prediccion.
    """
    confidence_percent = confidence_percent_from_result(result)
    return result.get("confidence_level") or get_confidence_level(confidence_percent)


def render_confidence_alert(confidence_percent: float) -> None:
    """
    Muestra una recomendacion visual segun el nivel de confianza.
    """
    level = get_confidence_level(confidence_percent)

    if level == "Alta confianza":
        st.success("Alta confianza: predicción confiable.")
    elif level == "Confianza media":
        st.warning("Confianza media: se recomienda revisar antes de guardar.")
    else:
        st.error(
            "Baja confianza: se recomienda corregir manualmente "
            "o no guardar automáticamente."
        )


def predict_batch_files(
    files_data: list | None = None,
    zip_data=None,
) -> dict[str, Any] | None:
    """
    Envía imágenes al endpoint /predict/batch y devuelve el resultado.
    """
    try:
        if files_data:
            files = [
                ("files", (f.name, f.getvalue(), f.type or "image/jpeg"))
                for f in files_data
            ]
            response = requests.post(
                f"{API_BASE_URL}/predict/batch",
                files=files,
                data=get_current_user(),
                timeout=300,
            )
        else:
            response = requests.post(
                f"{API_BASE_URL}/predict/batch",
                files={"zip": (zip_data.name, zip_data.getvalue(), "application/zip")},
                data=get_current_user(),
                timeout=300,
            )

        if response.status_code == 400:
            st.error(response.json().get("error", "Error en la solicitud."))
            return None

        if response.status_code != 200:
            st.error("Error al procesar el cargue masivo.")
            st.json(response.json())
            return None

        return response.json()

    except requests.RequestException as exc:
        st.error(f"No se pudo conectar con la API: {exc}")
        return None


def render_prediction_result(prediction_response: dict[str, Any]) -> None:
    """
    Muestra el resultado de predicción en pantalla.
    """
    prediction = prediction_response["prediction"]
    generated_code = prediction_response["generated_code"]

    predicted_category = prediction["predicted_category"]
    confidence_percent = confidence_percent_from_result(prediction)
    confidence_level = confidence_level_from_result(prediction)

    st.subheader("Resultado de clasificación")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Categoría", format_category(predicted_category))

    with col2:
        st.metric("Confianza", f"{confidence_percent:.2f}%")

    with col3:
        st.metric("Nivel de confianza", confidence_level)

    with col4:
        st.metric("Código generado", generated_code)

    render_confidence_alert(confidence_percent)

    st.write("Top predicciones:")

    top_predictions = prediction.get("top_predictions", [])

    top_data = [
        {
            "Categoría": format_category(item["category"]),
            "Confianza (%)": round(confidence_percent_from_result(item), 2),
        }
        for item in top_predictions[:3]
    ]

    st.dataframe(pd.DataFrame(top_data), use_container_width=True)


def render_inventory_table() -> None:
    """
    Muestra tabla de inventario.
    """
    audit_inventory_consult = bool(
        st.session_state.pop("audit_inventory_consult", False)
    )
    products = get_products(audit=audit_inventory_consult)

    st.subheader("Inventario registrado")

    if not products:
        st.info("Todavía no hay productos registrados.")
        return

    df = pd.DataFrame(products)

    if "category" in df.columns:
        df["category"] = df["category"].apply(format_category)

    if "confidence" in df.columns:
        df["confidence"] = (df["confidence"] * 100).round(2)

    columns_to_show = [
        "id",
        "code",
        "name",
        "category",
        "quantity",
        "confidence",
        "created_at",
        "status",
    ]

    existing_columns = [
        column for column in columns_to_show
        if column in df.columns
    ]

    st.dataframe(
        df[existing_columns],
        use_container_width=True,
    )


# ── Export helpers ───────────────────────────────────────────────────────────

def _xlsx_styles():
    from openpyxl.styles import PatternFill, Font, Alignment, Border, Side
    header_fill = PatternFill("solid", fgColor="E8611A")
    alt_fill    = PatternFill("solid", fgColor="F5F5F5")
    header_font = Font(color="FFFFFF", bold=True)
    data_font   = Font(color="091635")
    center      = Alignment(horizontal="center", vertical="center", wrap_text=True)
    thin        = Side(style="thin", color="D0D0D0")
    bdr         = Border(left=thin, right=thin, top=thin, bottom=thin)
    return header_fill, alt_fill, header_font, data_font, center, bdr


def _set_col_widths(ws, col_data: list[tuple[str, list]]) -> None:
    from openpyxl.utils import get_column_letter
    for ci, (header, values) in enumerate(col_data, start=1):
        max_len = max([len(str(header))] + [len(str(v)) for v in values], default=10)
        ws.column_dimensions[get_column_letter(ci)].width = min(max_len + 4, 50)


def _build_inventory_xlsx(flt: pd.DataFrame) -> bytes:
    from openpyxl import Workbook
    header_fill, alt_fill, header_font, data_font, center, bdr = _xlsx_styles()

    wb = Workbook()
    ws = wb.active
    ws.title = "Inventario"

    col_map = [
        ("id",        "ID"),
        ("code",      "Código"),
        ("name",      "Nombre"),
        ("category",  "Categoría"),
        ("quantity",  "Cantidad"),
        ("_conf_pct", "Confianza (%)"),
        ("_conf_level", "Nivel de confianza"),
        ("created_at","Fecha registro"),
        ("status",    "Estado"),
    ]
    existing = [(src, lbl) for src, lbl in col_map if src in flt.columns]

    ws.row_dimensions[1].height = 25
    for ci, (_, lbl) in enumerate(existing, start=1):
        cell = ws.cell(row=1, column=ci, value=lbl)
        cell.fill = header_fill
        cell.font = header_font
        cell.alignment = center
        cell.border = bdr

    df = flt.reset_index(drop=True)
    for ri, row in df.iterrows():
        row_fill = alt_fill if ri % 2 == 1 else None
        for ci, (src, _) in enumerate(existing, start=1):
            v = row.get(src, "")
            if src == "category":
                v = CATEGORY_LABELS.get(str(v), str(v))
            elif src == "_conf_pct":
                try:
                    v = round(float(v), 2)
                except (TypeError, ValueError):
                    pass
            elif src == "created_at":
                v = str(v).replace("T", " ").split(".")[0]
            cell = ws.cell(row=ri + 2, column=ci, value=v)
            cell.font = data_font
            cell.alignment = center
            cell.border = bdr
            if row_fill:
                cell.fill = row_fill

    col_data = []
    for src, lbl in existing:
        if src not in df.columns:
            col_data.append((lbl, []))
            continue
        raw = df[src].tolist()
        if src == "category":
            vals = [CATEGORY_LABELS.get(str(v), str(v)) for v in raw]
        elif src == "_conf_pct":
            vals = [f"{float(v):.2f}" if v != "" else "" for v in raw]
        elif src == "created_at":
            vals = [str(v).replace("T", " ").split(".")[0] for v in raw]
        else:
            vals = [str(v) for v in raw]
        col_data.append((lbl, vals))
    _set_col_widths(ws, col_data)

    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


def _build_batch_xlsx(ok_results: list[dict[str, Any]]) -> bytes:
    from openpyxl import Workbook
    header_fill, alt_fill, header_font, data_font, center, bdr = _xlsx_styles()

    wb = Workbook()
    ws = wb.active
    ws.title = "Clasificacion masiva"

    headers = ["Archivo", "Categoría", "Confianza (%)", "Nivel de confianza", "Top 2", "Top 3"]
    ws.row_dimensions[1].height = 25
    for ci, lbl in enumerate(headers, start=1):
        cell = ws.cell(row=1, column=ci, value=lbl)
        cell.fill = header_fill
        cell.font = header_font
        cell.alignment = center
        cell.border = bdr

    for ri, r in enumerate(ok_results):
        row_fill = alt_fill if ri % 2 == 1 else None
        tops = r.get("top_predictions", [])
        top2 = (
            f"{CATEGORY_LABELS.get(tops[1]['category'], tops[1]['category'])} "
            f"({tops[1]['confidence_percent']:.2f}%)"
            if len(tops) >= 2 else ""
        )
        top3 = (
            f"{CATEGORY_LABELS.get(tops[2]['category'], tops[2]['category'])} "
            f"({tops[2]['confidence_percent']:.2f}%)"
            if len(tops) >= 3 else ""
        )
        row_vals = [
            r.get("filename", ""),
            CATEGORY_LABELS.get(r.get("predicted_category", ""), r.get("predicted_category", "")),
            round(confidence_percent_from_result(r), 2),
            confidence_level_from_result(r),
            top2,
            top3,
        ]
        for ci, v in enumerate(row_vals, start=1):
            cell = ws.cell(row=ri + 2, column=ci, value=v)
            cell.font = data_font
            cell.alignment = center
            cell.border = bdr
            if row_fill:
                cell.fill = row_fill

    col_data = [
        ("Archivo",       [r.get("filename", "") for r in ok_results]),
        ("Categoría",     [CATEGORY_LABELS.get(r.get("predicted_category", ""), "") for r in ok_results]),
        ("Confianza (%)", [round(confidence_percent_from_result(r), 2) for r in ok_results]),
        ("Nivel de confianza", [confidence_level_from_result(r) for r in ok_results]),
        ("Top 2",         [""]),
        ("Top 3",         [""]),
    ]
    _set_col_widths(ws, col_data)

    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


def _build_inventory_pdf(
    flt: pd.DataFrame,
    total_prods: int,
    total_cats: int,
    total_stock: int,
    avg_conf: float,
) -> bytes:
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER, TA_LEFT
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.pdfgen import canvas as rl_canvas
    from reportlab.platypus import (
        HRFlowable, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
    )

    ORANGE     = colors.HexColor("#E8611A")
    LIGHT_GRAY = colors.HexColor("#F5F5F5")
    MUTED      = colors.HexColor("#69738a")
    DARK       = colors.HexColor("#091635")

    class _NumberedCanvas(rl_canvas.Canvas):
        def __init__(self, *args, **kwargs):
            rl_canvas.Canvas.__init__(self, *args, **kwargs)
            self._saved_page_states = []

        def showPage(self):
            self._saved_page_states.append(dict(self.__dict__))
            self._startPage()

        def save(self):
            num_pages = len(self._saved_page_states)
            for state in self._saved_page_states:
                self.__dict__.update(state)
                self._draw_footer(num_pages)
                rl_canvas.Canvas.showPage(self)
            rl_canvas.Canvas.save(self)

        def _draw_footer(self, total: int) -> None:
            self.saveState()
            self.setFont("Helvetica", 7.5)
            self.setFillColor(MUTED)
            self.drawString(
                2 * cm, 1.2 * cm,
                "Sistema Inteligente de Inventario — Reporte generado automaticamente",
            )
            self.drawRightString(
                A4[0] - 2 * cm, 1.2 * cm,
                f"Pagina {self._pageNumber} de {total}",
            )
            self.restoreState()

    buf       = io.BytesIO()
    timestamp = datetime.now().strftime("%d/%m/%Y a las %H:%M:%S")
    avail_w   = A4[0] - 4 * cm

    def _sty(name, **kw):
        return ParagraphStyle(name, **kw)

    title_sty = _sty("InvTitle", fontSize=18, fontName="Helvetica-Bold",
                     textColor=DARK, spaceAfter=2)
    sub_sty   = _sty("InvSub",   fontSize=12, fontName="Helvetica-Bold",
                     textColor=ORANGE, spaceAfter=2)
    date_sty  = _sty("InvDate",  fontSize=8.5, fontName="Helvetica", textColor=MUTED)
    mc_sty    = _sty("MC", fontSize=9, fontName="Helvetica",
                     textColor=DARK, alignment=TA_CENTER, leading=16)
    hdr_sty   = _sty("TH", fontSize=8, fontName="Helvetica-Bold",
                     textColor=colors.white, alignment=TA_CENTER)
    cell_sty  = _sty("TD", fontSize=7.5, fontName="Helvetica",
                     textColor=DARK, alignment=TA_CENTER)

    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        rightMargin=2*cm, leftMargin=2*cm,
        topMargin=2*cm, bottomMargin=2.5*cm,
        title="Reporte de Inventario",
    )

    story = []

    # Header
    story.append(Paragraph("Sistema Inteligente de Inventario", title_sty))
    story.append(Paragraph("Reporte de Inventario", sub_sty))
    story.append(Paragraph(f"Generado el {timestamp}", date_sty))
    story.append(Spacer(1, 0.3 * cm))
    story.append(HRFlowable(width="100%", thickness=2, lineCap="round", color=ORANGE))
    story.append(Spacer(1, 0.45 * cm))

    # Metrics row
    metrics_cells = [
        Paragraph(f"Productos registrados<br/><b>{total_prods}</b>", mc_sty),
        Paragraph(f"Categorias<br/><b>{total_cats}</b>", mc_sty),
        Paragraph(f"Stock total<br/><b>{total_stock}</b>", mc_sty),
        Paragraph(f"Confianza promedio<br/><b>{avg_conf:.2f}%</b>", mc_sty),
    ]
    metrics_tbl = Table([metrics_cells], colWidths=[avail_w / 4] * 4)
    metrics_tbl.setStyle(TableStyle([
        ("BACKGROUND",   (0, 0), (-1, -1), colors.HexColor("#FFF7E8")),
        ("BOX",          (0, 0), (-1, -1), 1.5, ORANGE),
        ("INNERGRID",    (0, 0), (-1, -1), 0.5, ORANGE),
        ("ALIGN",        (0, 0), (-1, -1), "CENTER"),
        ("VALIGN",       (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING",   (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 10),
    ]))
    story.append(metrics_tbl)
    story.append(Spacer(1, 0.6 * cm))

    # Main inventory table
    col_map = [
        ("id",        "ID"),
        ("code",      "Codigo"),
        ("name",      "Nombre"),
        ("category",  "Categoria"),
        ("quantity",  "Cantidad"),
        ("_conf_pct", "Confianza (%)"),
        ("_conf_level", "Nivel"),
        ("created_at","Fecha"),
        ("status",    "Estado"),
    ]
    existing = [(src, lbl) for src, lbl in col_map if src in flt.columns]

    table_data = [[Paragraph(lbl, hdr_sty) for _, lbl in existing]]
    for _, row in flt.reset_index(drop=True).iterrows():
        row_cells = []
        for src, _ in existing:
            v = row.get(src, "")
            if src == "category":
                v = CATEGORY_LABELS.get(str(v), str(v))
            elif src == "_conf_pct":
                try:
                    v = f"{float(v):.2f}%"
                except (TypeError, ValueError):
                    pass
            elif src == "created_at":
                v = str(v).replace("T", " ").split(".")[0]
            row_cells.append(Paragraph(str(v), cell_sty))
        table_data.append(row_cells)

    width_map = {
        "id": 0.5, "code": 1.2, "name": 2.0, "category": 1.8,
        "quantity": 0.8, "_conf_pct": 1.2, "_conf_level": 1.5,
        "created_at": 1.8, "status": 0.9,
    }
    raw_w = [width_map.get(src, 1.0) for src, _ in existing]
    col_widths_pdf = [avail_w * w / sum(raw_w) for w in raw_w]

    main_tbl = Table(table_data, colWidths=col_widths_pdf, repeatRows=1)
    main_tbl.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0),  ORANGE),
        ("TEXTCOLOR",     (0, 0), (-1, 0),  colors.white),
        ("FONTNAME",      (0, 0), (-1, 0),  "Helvetica-Bold"),
        ("FONTSIZE",      (0, 0), (-1, 0),  8),
        ("ALIGN",         (0, 0), (-1, -1), "CENTER"),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
        ("GRID",          (0, 0), (-1, -1), 0.4, colors.HexColor("#D0D0D0")),
        ("TOPPADDING",    (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1), [colors.white, LIGHT_GRAY]),
    ]))
    story.append(main_tbl)

    doc.build(story, canvasmaker=_NumberedCanvas)
    buf.seek(0)
    return buf.getvalue()


# ── Per-category badge colors ─────────────────────────────────────────────────
_CAT_CHIP_COLORS: dict[str, tuple[str, str]] = {
    "arroz_y_granos":       ("#fff7e8", "#d97706"),
    "pastas":               ("#eaf3ff", "#2563eb"),
    "aceites":              ("#ecfdf3", "#16a34a"),
    "salsas_y_condimentos": ("#fdf2f8", "#9333ea"),
    "cafe_chocolate":       ("#fff1ef", "#dc2626"),
    "enlatados":            ("#f0f9ff", "#0284c7"),
    "azucar_sal":           ("#fdf4ff", "#7c3aed"),
}

_SHARED_CSS = """
<style>
.inv-metric-card{background:#fff;border:1px solid var(--panel-border);border-radius:8px;
  padding:1rem .9rem;display:flex;align-items:center;justify-content:space-between;min-height:80px}
.inv-metric-label{color:var(--muted);font-size:.78rem;font-weight:700;margin-bottom:.25rem}
.inv-metric-value{color:var(--ink);font-size:1.35rem;font-weight:800}
.inv-metric-icon{width:40px;height:40px;background:#fff7e8;color:#f59e0b;border-radius:8px;
  display:inline-flex;align-items:center;justify-content:center;font-size:1.15rem;flex:0 0 auto}
.cat-badge{border-radius:999px;display:inline-flex;align-items:center;
  font-size:.75rem;font-weight:800;padding:.2rem .65rem;white-space:nowrap}
.conf-bar-wrap{display:flex;align-items:center;gap:.45rem}
.conf-bar-track{background:#e8edf5;border-radius:999px;height:8px;flex:1;min-width:55px;overflow:hidden}
.conf-bar-fill{background:linear-gradient(90deg,#22c55e,#86efac);border-radius:inherit;height:100%}
.conf-value{font-size:.8rem;font-weight:800;color:var(--ink);min-width:44px;text-align:right}
.inv-status-ok{background:#ecfdf3;color:#15803d;border-radius:999px;
  display:inline-flex;align-items:center;gap:.3rem;font-size:.75rem;font-weight:800;padding:.2rem .65rem}
.inv-table{width:100%;border-collapse:collapse;font-size:.875rem}
.inv-table th{background:#f8fafc;color:#526079;font-size:.75rem;font-weight:800;
  padding:.6rem .75rem;text-align:left;border-bottom:1px solid var(--panel-border)}
.inv-table td{padding:.6rem .75rem;border-bottom:1px solid #f0f4fa;color:var(--ink);vertical-align:middle}
.inv-table tr:last-child td{border-bottom:0}
.inv-table tr:hover td{background:#f8fafc}
.inv-banner{background:linear-gradient(90deg,#eff7ff,#f7fbff);border:1px solid #bfdbfe;
  border-radius:8px;padding:1.25rem 1.5rem;display:flex;align-items:center;
  justify-content:space-between;margin-top:1.25rem;gap:1rem}
.inv-banner-title{color:var(--ink);font-weight:800;font-size:1rem;margin-bottom:.3rem}
.inv-banner-copy{color:var(--muted);font-size:.88rem}
.inv-banner-icon{font-size:3rem;opacity:.22;flex:0 0 auto}
.mdl-subtitle{color:var(--muted);font-size:.95rem;margin:-.5rem 0 1.25rem}
.mdl-confidence-copy{background:#fff;border:1px solid var(--panel-border);border-radius:8px;
  color:var(--muted);font-size:.9rem;line-height:1.55;padding:.9rem 1rem}
.mdl-status-ok{color:#15803d;font-weight:800}
.mdl-status-err{color:#dc2626;font-weight:800}
.mdl-detail-row{display:flex;align-items:flex-start;gap:.65rem;padding:.5rem 0;
  border-bottom:1px solid #f0f4fa;font-size:.88rem}
.mdl-detail-row:last-child{border-bottom:0}
.mdl-detail-icon{flex:0 0 auto;width:24px;height:24px;background:#fff7e8;color:#f59e0b;
  border-radius:6px;display:inline-flex;align-items:center;justify-content:center;font-size:.82rem}
.mdl-detail-label{color:var(--muted);font-weight:700;min-width:130px;flex:0 0 auto}
.mdl-detail-value{color:var(--ink);font-weight:700}
.mdl-file-row{display:flex;align-items:center;gap:.65rem;padding:.5rem 0;
  border-bottom:1px solid #f0f4fa}
.mdl-file-row:last-child{border-bottom:0}
.mdl-file-icon{flex:0 0 auto;width:28px;height:28px;background:#eaf3ff;color:#2563eb;
  border-radius:6px;display:inline-flex;align-items:center;justify-content:center;font-size:.85rem}
.mdl-file-path{color:var(--ink);font-weight:700;font-family:monospace;font-size:.83rem;
  flex:1;min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
</style>
"""


def _metric_card(label: str, value: str, icon: str, value_html: str = "") -> str:
    val_part = value_html if value_html else escape(value)
    return (
        f'<div class="inv-metric-card">'
        f'<div><div class="inv-metric-label">{escape(label)}</div>'
        f'<div class="inv-metric-value">{val_part}</div></div>'
        f'<div class="inv-metric-icon">{icon}</div>'
        f'</div>'
    )


def _cat_badge(cat_key: str, label: str) -> str:
    bg, fg = _CAT_CHIP_COLORS.get(cat_key, ("#f3f6fb", "#526079"))
    return (
        f'<span class="cat-badge" style="background:{bg};color:{fg}">'
        f'{escape(label)}</span>'
    )


def _conf_bar(pct: float) -> str:
    pct = min(max(pct, 0.0), 100.0)
    return (
        f'<div class="conf-bar-wrap">'
        f'<div class="conf-bar-track">'
        f'<div class="conf-bar-fill" style="width:{pct:.1f}%"></div>'
        f'</div>'
        f'<span class="conf-value">{pct:.2f}%</span>'
        f'</div>'
    )


def render_inventory_tab() -> None:
    st.markdown(_SHARED_CSS, unsafe_allow_html=True)
    st.markdown(
        '<div class="panel-title">'
        '<span>▣</span><span>Inventario registrado</span>'
        '<span style="font-size:.9rem;margin-left:.5rem;color:var(--muted)">⌗</span>'
        '</div>',
        unsafe_allow_html=True,
    )

    audit_inventory_consult = bool(
        st.session_state.pop("audit_inventory_consult", False)
    )
    products = get_products(audit=audit_inventory_consult)

    if not products:
        st.info("Todavía no hay productos registrados.")
        return

    df = pd.DataFrame(products)

    if "confidence" in df.columns:
        df["_conf_pct"] = df["confidence"].apply(normalize_confidence_for_display).round(2)
        confidence_metrics = calculate_confidence_metrics(df["confidence"].tolist())
    else:
        df["_conf_pct"] = 0.0
        confidence_metrics = calculate_confidence_metrics([])

    df["_conf_level"] = df["_conf_pct"].apply(get_confidence_level)

    # ── Metrics ───────────────────────────────────────────────────────────
    total_prods = len(df)
    total_cats  = df["category"].nunique() if "category" in df.columns else 0
    total_stock = int(df["quantity"].sum()) if "quantity" in df.columns else 0
    avg_conf    = float(confidence_metrics.get("average_confidence", 0.0))

    mc1, mc2, mc3, mc4 = st.columns(4)
    for col, lbl, val, icon in [
        (mc1, "Productos registrados", str(total_prods),   "▣"),
        (mc2, "Categorías",            str(total_cats),    "◈"),
        (mc3, "Stock total",           str(total_stock),   "▤"),
        (mc4, "Confianza promedio",    f"{avg_conf:.2f}%", "◎"),
    ]:
        with col:
            st.markdown(_metric_card(lbl, val, icon), unsafe_allow_html=True)

    st.markdown("<div style='height:.75rem'></div>", unsafe_allow_html=True)
    render_confidence_metric_cards(
        confidence_metrics,
        include_total=False,
        include_average=False,
    )

    st.markdown("<div style='height:.75rem'></div>", unsafe_allow_html=True)

    # ── Filters ───────────────────────────────────────────────────────────
    cats_list   = sorted(df["category"].unique().tolist()) if "category" in df.columns else []
    status_list = sorted(df["status"].unique().tolist())   if "status"   in df.columns else []
    cat_labels  = ["Todas las categorías"] + [CATEGORY_LABELS.get(c, c) for c in cats_list]
    status_opts = ["Todos los estados"] + status_list

    f1, f2, f3 = st.columns([2.5, 1.6, 1.6])
    with f1:
        search = st.text_input(
            "search_inv",
            placeholder="⌕  Buscar por nombre o código...",
            label_visibility="collapsed",
        )
    with f2:
        cat_sel = st.selectbox("cat_inv", cat_labels, label_visibility="collapsed")
    with f3:
        status_sel = st.selectbox("status_inv", status_opts, label_visibility="collapsed")

    # ── Apply filters ─────────────────────────────────────────────────────
    flt = df.copy()
    if search:
        mask = pd.Series(False, index=flt.index)
        for col in ("name", "code"):
            if col in flt.columns:
                mask |= flt[col].astype(str).str.contains(search, case=False, na=False)
        flt = flt[mask]
    if cat_sel != "Todas las categorías":
        inv_map = {v: k for k, v in CATEGORY_LABELS.items()}
        cat_key = inv_map.get(cat_sel, cat_sel)
        if "category" in flt.columns:
            flt = flt[flt["category"] == cat_key]
    if status_sel != "Todos los estados" and "status" in flt.columns:
        flt = flt[flt["status"] == status_sel]

    # ── Action buttons ────────────────────────────────────────────────────
    _ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    is_admin = current_user_role() == "admin"
    b1, b2, b3, _sp = st.columns([1.5, 1.4, 1.3, 2.8])
    with b1:
        if st.button("↻  Actualizar inventario", type="primary", use_container_width=True):
            st.session_state["audit_inventory_consult"] = True
            st.rerun()
    with b2:
        st.download_button(
            "⬇  Exportar Excel",
            data=_build_inventory_xlsx(flt),
            file_name=f"inventario_{_ts}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
            key="btn_inv_xlsx",
            disabled=not is_admin,
        )
    with b3:
        st.download_button(
            "⬇  Exportar PDF",
            data=_build_inventory_pdf(flt, total_prods, total_cats, total_stock, avg_conf),
            file_name=f"reporte_inventario_{_ts}.pdf",
            mime="application/pdf",
            use_container_width=True,
            key="btn_inv_pdf",
            disabled=not is_admin,
        )
    if not is_admin:
        st.caption("La exportacion de inventario esta disponible solo para admin.")

    # ── Table ──────────────────────────────────────────────────────────────
    display_cols = [
        "id", "code", "name", "category", "quantity",
        "_conf_pct", "_conf_level", "created_at", "status",
    ]
    col_labels   = {
        "id": "#", "code": "Código", "name": "Nombre", "category": "Categoría",
        "quantity": "Cantidad", "_conf_pct": "Confianza",
        "_conf_level": "Nivel de confianza",
        "created_at": "Fecha", "status": "Estado",
    }
    visible = [c for c in display_cols if c in flt.columns]

    # --- NUEVO: LÓGICA DE PAGINACIÓN ---
    ITEMS_PER_PAGE = 10
    total_items = len(flt)
    total_pages = max(1, (total_items + ITEMS_PER_PAGE - 1) // ITEMS_PER_PAGE)

    # Inicializar la página actual en memoria si no existe
    if "inventory_page" not in st.session_state:
        st.session_state.inventory_page = 1

    # Si se aplica un filtro y la página actual queda vacía, la devolvemos a la 1
    if st.session_state.inventory_page > total_pages:
        st.session_state.inventory_page = 1

    current_page = st.session_state.inventory_page
    start_idx = (current_page - 1) * ITEMS_PER_PAGE
    end_idx = start_idx + ITEMS_PER_PAGE

    # Cortamos la tabla para que solo tenga los 10 registros de la página actual
    paginated_flt = flt.iloc[start_idx:end_idx]

    thead = "".join(f'<th>{escape(col_labels.get(c, c))}</th>' for c in visible)
    tbody = ""
    
    # Renderizamos únicamente la porción de la tabla seleccionada
    for row_i, row in paginated_flt.reset_index(drop=True).iterrows():
        cells = ""
        for col in visible:
            v = row.get(col, "")
            if col == "category":
                k = str(v)
                cells += f"<td>{_cat_badge(k, CATEGORY_LABELS.get(k, k))}</td>"
            elif col == "_conf_pct":
                try:
                    cells += f"<td>{_conf_bar(float(v))}</td>"
                except (TypeError, ValueError):
                    cells += f"<td>{escape(str(v))}</td>"
            elif col == "_conf_level":
                cells += f"<td>{render_confidence_level_badge(str(v))}</td>"
            elif col == "status":
                cells += f'<td><span class="inv-status-ok">● {escape(str(v))}</span></td>'
            elif col == "id":
                # Aseguramos que la numeración sea continua (1..10, 11..20, etc.)
                cells += f"<td>{start_idx + row_i + 1}</td>"
            else:
                cells += f"<td>{escape(str(v))}</td>"
        tbody += f"<tr>{cells}</tr>"

    if not tbody:
        tbody = (
            f'<tr><td colspan="{len(visible)}" '
            f'style="text-align:center;color:var(--muted);padding:2rem">'
            f'Sin resultados para los filtros aplicados.</td></tr>'
        )

    st.markdown(
        f'<div style="overflow-x:auto;margin-top:.75rem;border:1px solid var(--panel-border);'
        f'border-radius:8px;overflow:hidden">'
        f'<table class="inv-table"><thead><tr>{thead}</tr></thead>'
        f'<tbody>{tbody}</tbody></table></div>',
        unsafe_allow_html=True,
    )

    # --- CONTROLES DE PAGINACIÓN VISUALES ---
    if total_pages > 1:
        st.markdown("<div style='height: 0.5rem'></div>", unsafe_allow_html=True)
        col_prev, col_info, col_next = st.columns([1, 4, 1])
        
        with col_prev:
            if st.button("⬅️ Anterior", disabled=current_page == 1, use_container_width=True):
                st.session_state.inventory_page -= 1
                st.rerun()
                
        with col_info:
            st.markdown(
                f"<div style='text-align: center; padding-top: 0.4rem; color: var(--muted); font-size: 0.95rem;'>"
                f"Página <b>{current_page}</b> de <b>{total_pages}</b>"
                f"</div>",
                unsafe_allow_html=True
            )
            
        with col_next:
            if st.button("Siguiente ➡️", disabled=current_page == total_pages, use_container_width=True):
                st.session_state.inventory_page += 1
                st.rerun()

    # ── Bottom banner ──────────────────────────────────────────────────────
    if not flt.empty and "code" in flt.columns:
        with st.container(border=True):
            render_small_panel_title("Editar producto", "✎")
            selected_code = st.selectbox(
                "Selecciona el código del producto a editar",
                options=flt["code"].astype(str).tolist(),
                key="admin_product_code",
            )
            selected_rows = df[df["code"].astype(str) == selected_code]
            selected_row = selected_rows.iloc[0] if not selected_rows.empty else None

            if selected_row is not None:
                # Obtenemos la categoría original para enviarla oculta
                current_category = str(selected_row.get("category", ""))
                
                # Mostramos la categoría solo como texto informativo
                st.caption(f"**Categoría asignada:** {format_category(current_category)} *(No editable)*")

                # Dejamos solo 2 columnas: Nombre y Cantidad
                edit_col_1, edit_col_2 = st.columns([2, 1])
                with edit_col_1:
                    edit_name = st.text_input(
                        "Nombre",
                        value=str(selected_row.get("name", "")),
                        # CLAVE DINÁMICA: Cambia según el código seleccionado
                        key=f"edit_name_{selected_code}", 
                    )
                with edit_col_2:
                    edit_quantity = st.number_input(
                        "Cantidad",
                        min_value=0,
                        value=int(selected_row.get("quantity", 0)),
                        step=1,
                        # CLAVE DINÁMICA: Cambia según el código seleccionado
                        key=f"edit_qty_{selected_code}", 
                    )

                save_col, deactivate_col = st.columns(2)
                with save_col:
                    if st.button(
                        "Actualizar producto",
                        type="primary",
                        use_container_width=True,
                    ):
                        updated = update_product_api(
                            selected_code,
                            {
                                "name": edit_name,
                                "quantity": int(edit_quantity),
                                "category": current_category, # Enviamos la categoría intacta
                                "audit_details": "Actualización de nombre/cantidad desde Streamlit.",
                            },
                        )
                        if updated:
                            st.success("Producto actualizado correctamente.")
                            st.rerun()

                with deactivate_col:
                    # Dejamos la opción de desactivar solo para admins por seguridad
                    if is_admin:
                        if st.button(
                            "Desactivar producto",
                            use_container_width=True,
                        ):
                            deactivated = deactivate_product_api(selected_code)
                            if deactivated:
                                st.success("Producto desactivado.")
                                st.rerun()
                    else:
                        st.button(
                            "Desactivar producto",
                            use_container_width=True,
                            disabled=True,
                            help="Solo los administradores pueden desactivar/eliminar productos del sistema."
                        )

    st.markdown(
        '<div class="inv-banner">'
        '<div>'
        '<div class="inv-banner-title">Gestiona y revisa tu inventario</div>'
        '<div class="inv-banner-copy">Aquí puedes consultar los productos detectados, '
        'revisar su información y mantener tu inventario actualizado.</div>'
        '</div>'
        '<div class="inv-banner-icon">▣▣▣</div>'
        '</div>',
        unsafe_allow_html=True,
    )


def _format_audit_confidence(value: Any) -> str:
    if value in (None, ""):
        return "-"

    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return str(value)

    if confidence <= 1:
        confidence *= 100

    return f"{confidence:.2f}%"


def _format_audit_details(value: Any) -> str:
    if value in (None, ""):
        return "-"

    if not isinstance(value, str):
        return str(value)

    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return value

    if isinstance(parsed, dict):
        return "; ".join(
            f"{key}: {parsed_value}"
            for key, parsed_value in parsed.items()
            if parsed_value not in (None, "")
        ) or "-"

    return str(parsed)


def render_logs_tab() -> None:
    """
    Muestra el historial de auditoria del inventario.
    """
    st.markdown(_SHARED_CSS, unsafe_allow_html=True)
    render_panel_title("Historial de actividad", "☷")
    st.caption(
        "Consulta las acciones realizadas sobre el inventario y los productos clasificados."
    )

    with st.container(border=True):
        render_small_panel_title("Filtros", "⌕")

        # Ajustamos a 3 columnas ya que eliminamos el selector de acciones
        filter_col_1, filter_col_2, filter_col_3 = st.columns(
            [1.5, 1.5, 1.0],
            gap="medium",
        )

        with filter_col_1:
            user_filter = st.text_input(
                "Filtro por usuario/correo",
                placeholder="Nombre o correo",
                key="audit_user_filter",
            )

        with filter_col_2:
            product_code_filter = st.text_input(
                "Filtro por codigo de producto",
                placeholder="INV-...",
                key="audit_product_code_filter",
            )

        with filter_col_3:
            # Espaciador para que el botón quede alineado con los inputs de texto
            st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
            refresh_clicked = st.button(
                "Actualizar logs",
                type="primary",
                use_container_width=True,
            )

    if refresh_clicked:
        st.session_state["audit_logs_refreshed_at"] = datetime.now().isoformat()

    # AQUÍ ESTÁ LA MODIFICACIÓN CLAVE: 
    # 1. Preparamos los filtros base (usuario y código de producto)
    base_filters: dict[str, str] = {}
    
    cleaned_user_filter = user_filter.strip()
    if cleaned_user_filter:
        if "@" in cleaned_user_filter:
            base_filters["user_email"] = cleaned_user_filter
        else:
            base_filters["user_name"] = cleaned_user_filter

    cleaned_product_code = product_code_filter.strip()
    if cleaned_product_code:
        base_filters["product_code"] = cleaned_product_code

    # 2. Consultamos los tres movimientos clave del inventario
    logs_agregados = get_audit_logs({**base_filters, "action": "producto_agregado"})
    logs_actualizados = get_audit_logs({**base_filters, "action": "producto_actualizado"})
    logs_desactivados = get_audit_logs({**base_filters, "action": "producto_desactivado"})

    # 3. Unimos los resultados en una sola lista
    logs = logs_agregados + logs_actualizados + logs_desactivados
    
    # 4. Ordenamos todo por fecha (del más reciente al más antiguo)
    logs.sort(key=lambda x: str(x.get("created_at", "")), reverse=True)

    total_logs = len(logs)
    total_users = len(
        {
            log.get("user_email") or log.get("user_name")
            for log in logs
            if log.get("user_email") or log.get("user_name")
        }
    )
    total_actions = len(
        {
            log.get("action")
            for log in logs
            if log.get("action")
        }
    )

    metric_col_1, metric_col_2, metric_col_3 = st.columns(3)
    with metric_col_1:
        st.markdown(
            _metric_card("Registros encontrados", str(total_logs), "☷"),
            unsafe_allow_html=True,
        )
    with metric_col_2:
        st.markdown(
            _metric_card("Usuarios", str(total_users), "◉"),
            unsafe_allow_html=True,
        )
    with metric_col_3:
        st.markdown(
            _metric_card("Tipos de accion", str(total_actions), "↻"),
            unsafe_allow_html=True,
        )

    if not logs:
        st.info("No hay logs para los filtros seleccionados.")
        return

    rows = []
    for log in logs:
        created_at = str(log.get("created_at") or "")
        created_at = created_at.replace("T", " ").split(".")[0]

        rows.append(
            {
                "Fecha": created_at,
                "Usuario": log.get("user_name") or "-",
                "Correo": log.get("user_email") or "-",
                "Rol": log.get("user_role") or "-",
                "Accion": format_audit_action(log.get("action")),
                "Codigo producto": log.get("product_code") or "-",
                "Producto": log.get("product_name") or "-",
                "Categoria": format_category(log.get("category") or "-"),
                "Cantidad": log.get("quantity") if log.get("quantity") is not None else "-",
                "Confianza": _format_audit_confidence(log.get("confidence")),
                "Detalles": _format_audit_details(log.get("details")),
            }
        )

    st.dataframe(
        pd.DataFrame(rows),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown(
        '<div class="inv-banner">'
        '<div>'
        '<div class="inv-banner-title">Trazabilidad del inventario</div>'
        '<div class="inv-banner-copy">Cada registro conserva usuario, accion, producto, '
        'fecha y detalles operativos de la actividad.</div>'
        '</div>'
        '<div class="inv-banner-icon">☷</div>'
        '</div>',
        unsafe_allow_html=True,
    )


def render_users_tab() -> None:
    """
    Administra usuarios de la aplicacion. Solo admin.
    """
    if current_user_role() != "admin":
        st.warning("Solo los administradores pueden administrar usuarios.")
        return

    st.markdown(_SHARED_CSS, unsafe_allow_html=True)
    render_panel_title("Usuarios", "o")

    left_col, right_col = st.columns([1, 1.35], gap="medium")

    with left_col:
        with st.container(border=True):
            render_small_panel_title("Crear usuario", "+")
            with st.form("create_user_form"):
                full_name = st.text_input("Nombre completo")
                email = st.text_input("Correo")
                password = st.text_input("Contrasena", type="password")
                role = st.selectbox("Rol", ["bodega", "admin"])
                submitted = st.form_submit_button(
                    "Crear usuario",
                    type="primary",
                    use_container_width=True,
                )

            if submitted:
                created_user = create_app_user(
                    {
                        "full_name": full_name,
                        "email": email,
                        "password": password,
                        "role": role,
                        "status": "activo",
                    }
                )
                if created_user:
                    st.success("Usuario creado correctamente.")
                    st.rerun()

    with right_col:
        users = get_users()
        if not users:
            st.info("No hay usuarios registrados.")
            return

        rows = []
        for user in users:
            rows.append(
                {
                    "ID": user.get("id"),
                    "Nombre": user.get("full_name"),
                    "Correo": user.get("email"),
                    "Rol": user.get("role"),
                    "Estado": user.get("status"),
                    "Creado": str(user.get("created_at") or "").replace("T", " ").split(".")[0],
                }
            )

        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

def render_model_results_tab() -> None:
    """
    Muestra los resultados de evaluación del modelo:
    accuracy, correctas, incorrectas, falsos positivos, falsos negativos,
    matriz de confusión, métricas por clase y distribución de predicciones.
    """
    st.markdown(_SHARED_CSS, unsafe_allow_html=True)

    report_path = Path("trained_models/evaluation_report.json")
    confusion_matrix_path = Path("trained_models/confusion_matrix.png")
    metrics_by_class_path = Path("trained_models/test_metrics_by_class.png")
    prediction_distribution_path = Path("trained_models/prediction_distribution.png")
    test_predictions_path = Path("trained_models/test_predictions.csv")

    render_panel_title("Resultados del modelo", "▧")

    st.markdown(
        '<div class="mdl-subtitle">'
        'Evaluación del rendimiento del modelo usando el conjunto de prueba. '
        'Aquí se muestran métricas reales como accuracy, predicciones correctas, '
        'errores, falsos positivos, falsos negativos y gráficos de desempeño.'
        '</div>',
        unsafe_allow_html=True,
    )

    if not report_path.exists():
        st.warning(
            "No se encontró el archivo de evaluación del modelo. "
            "Ejecuta primero el script:"
        )
        st.code("python scripts/evaluate_model.py", language="bash")

        st.markdown(
            '<div class="inv-banner">'
            '<div>'
            '<div class="inv-banner-title">Resultados pendientes de generar</div>'
            '<div class="inv-banner-copy">'
            'Cuando ejecutes la evaluación, se crearán archivos como '
            'evaluation_report.json, confusion_matrix.png, '
            'test_metrics_by_class.png y prediction_distribution.png.'
            '</div>'
            '</div>'
            '<div class="inv-banner-icon">▤</div>'
            '</div>',
            unsafe_allow_html=True,
        )
        return

    try:
        evaluation_report = json.loads(report_path.read_text(encoding="utf-8"))
    except Exception as exc:
        st.error(f"No se pudo leer evaluation_report.json: {exc}")
        return

    total_images = int(evaluation_report.get("total_images", 0))
    correct_predictions = int(evaluation_report.get("correct_predictions", 0))
    incorrect_predictions = int(evaluation_report.get("incorrect_predictions", 0))
    test_accuracy_percent = float(evaluation_report.get("test_accuracy_percent", 0))
    average_confidence_percent = float(
        evaluation_report.get("average_confidence_percent", 0)
    )
    false_positives_total = int(evaluation_report.get("false_positives_total", 0))
    false_negatives_total = int(evaluation_report.get("false_negatives_total", 0))

    # ── Tarjetas principales ───────────────────────────────────────────────
    metric_col_1, metric_col_2, metric_col_3, metric_col_4 = st.columns(4)

    with metric_col_1:
        st.markdown(
            _metric_card("Accuracy test", f"{test_accuracy_percent:.2f}%", "◎"),
            unsafe_allow_html=True,
        )

    with metric_col_2:
        st.markdown(
            _metric_card("Imágenes evaluadas", str(total_images), "▤"),
            unsafe_allow_html=True,
        )

    with metric_col_3:
        st.markdown(
            _metric_card("Correctas", str(correct_predictions), "✓"),
            unsafe_allow_html=True,
        )

    with metric_col_4:
        st.markdown(
            _metric_card("Incorrectas", str(incorrect_predictions), "!"),
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height:.75rem'></div>", unsafe_allow_html=True)

    metric_col_5, metric_col_6, metric_col_7 = st.columns(3)

    with metric_col_5:
        st.markdown(
            _metric_card("Falsos positivos", str(false_positives_total), "+"),
            unsafe_allow_html=True,
        )

    with metric_col_6:
        st.markdown(
            _metric_card("Falsos negativos", str(false_negatives_total), "-"),
            unsafe_allow_html=True,
        )

    with metric_col_7:
        st.markdown(
            _metric_card(
                "Confianza promedio",
                f"{average_confidence_percent:.2f}%",
                "⌁",
            ),
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

    # ── Explicación rápida ────────────────────────────────────────────────
    with st.container(border=True):
        render_small_panel_title("Resumen de evaluación", "ⓘ")
        st.markdown(
            """
            Estas métricas se calculan comparando la categoría real de cada imagen del
            conjunto de prueba contra la categoría predicha por el modelo.

            - **Accuracy:** porcentaje total de imágenes clasificadas correctamente.
            - **Correctas:** imágenes donde la categoría real coincide con la predicción.
            - **Incorrectas:** imágenes donde el modelo se equivocó.
            - **Falsos positivos:** veces que una categoría fue predicha incorrectamente.
            - **Falsos negativos:** veces que una categoría real no fue detectada correctamente.
            """
        )

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

    # ── Gráficos principales ───────────────────────────────────────────────
    graph_col_1, graph_col_2 = st.columns(2, gap="medium")

    with graph_col_1:
        with st.container(border=True):
            render_small_panel_title("Matriz de confusión", "▦")

            if confusion_matrix_path.exists():
                st.image(
                    str(confusion_matrix_path),
                    caption="Relación entre clases reales y clases predichas.",
                    use_container_width=True,
                )
            else:
                st.info("No se encontró confusion_matrix.png.")

    with graph_col_2:
        with st.container(border=True):
            render_small_panel_title("Métricas por categoría", "▥")

            if metrics_by_class_path.exists():
                st.image(
                    str(metrics_by_class_path),
                    caption="Precision, recall y F1-score por categoría.",
                    use_container_width=True,
                )
            else:
                st.info("No se encontró test_metrics_by_class.png.")

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

    with st.container(border=True):
        render_small_panel_title("Distribución de predicciones", "▤")

        if prediction_distribution_path.exists():
            st.image(
                str(prediction_distribution_path),
                caption="Cantidad de imágenes predichas por cada categoría.",
                use_container_width=True,
            )
        else:
            st.info("No se encontró prediction_distribution.png.")

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

    # ── Tabla por categoría ────────────────────────────────────────────────
    per_class_metrics = evaluation_report.get("per_class_metrics", {})

    if per_class_metrics:
        rows = []

        for category, metrics in per_class_metrics.items():
            rows.append(
                {
                    "Categoría": format_category(category),
                    "Precision": round(float(metrics.get("precision", 0)) * 100, 2),
                    "Recall": round(float(metrics.get("recall", 0)) * 100, 2),
                    "F1-score": round(float(metrics.get("f1_score", 0)) * 100, 2),
                    "Soporte": int(metrics.get("support", 0)),
                    "Verdaderos positivos": int(metrics.get("true_positives", 0)),
                    "Falsos positivos": int(metrics.get("false_positives", 0)),
                    "Falsos negativos": int(metrics.get("false_negatives", 0)),
                }
            )

        with st.container(border=True):
            render_small_panel_title("Detalle por categoría", "☷")
            st.dataframe(
                pd.DataFrame(rows),
                use_container_width=True,
                hide_index=True,
            )

    # ── Predicciones del test ──────────────────────────────────────────────
    if test_predictions_path.exists():
        with st.expander("Ver predicciones individuales del test"):
            try:
                predictions_df = pd.read_csv(test_predictions_path)

                columns_to_show = [
                    "image_path",
                    "true_label",
                    "predicted_label",
                    "confidence_percent",
                    "is_correct",
                ]

                existing_columns = [
                    column
                    for column in columns_to_show
                    if column in predictions_df.columns
                ]

                if "true_label" in predictions_df.columns:
                    predictions_df["true_label"] = predictions_df["true_label"].apply(
                        format_category
                    )

                if "predicted_label" in predictions_df.columns:
                    predictions_df["predicted_label"] = predictions_df[
                        "predicted_label"
                    ].apply(format_category)

                if "confidence_percent" in predictions_df.columns:
                    predictions_df["confidence_percent"] = predictions_df[
                        "confidence_percent"
                    ].round(2)

                st.dataframe(
                    predictions_df[existing_columns],
                    use_container_width=True,
                    hide_index=True,
                )

            except Exception as exc:
                st.warning(f"No se pudo leer test_predictions.csv: {exc}")

    st.markdown(
        '<div class="inv-banner">'
        '<div>'
        '<div class="inv-banner-title">Evaluación del modelo completada</div>'
        '<div class="inv-banner-copy">'
        'Estos resultados permiten analizar el desempeño real del modelo sobre '
        'imágenes de prueba y detectar categorías donde presenta más errores.'
        '</div>'
        '</div>'
        '<div class="inv-banner-icon">◎▧</div>'
        '</div>',
        unsafe_allow_html=True,
    )

def render_model_tab() -> None:
    report_path = Path("trained_models/training_report.json")
    report_ok   = report_path.exists()

    st.markdown(_SHARED_CSS, unsafe_allow_html=True)

    # ── Title + subtitle ──────────────────────────────────────────────────
    st.markdown(
        '<div class="panel-title"><span>ⓘ</span><span>Información del modelo</span></div>'
        '<div class="mdl-subtitle">Arquitectura, categorías y archivos del modelo de clasificación.</div>',
        unsafe_allow_html=True,
    )

    # ── 4 metric cards ─────────────────────────────────────────────────────
    status_html = (
        '<span class="mdl-status-ok">● Disponible</span>'
        if report_ok
        else '<span class="mdl-status-err">● No disponible</span>'
    )
    mc1, mc2, mc3, mc4 = st.columns(4)
    for col, lbl, val, icon in [
        (mc1, "Arquitectura",       "MobileNetV2",  "◎"),
        (mc2, "Categorías",         "7",            "◈"),
        (mc3, "Formato del modelo", "Keras / ONNX", "▤"),
    ]:
        with col:
            st.markdown(_metric_card(lbl, val, icon), unsafe_allow_html=True)
    with mc4:
        st.markdown(
            _metric_card("Estado del reporte", "", "▣", value_html=status_html),
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

    # ── Two columns ────────────────────────────────────────────────────────
    left_col, right_col = st.columns([1, 1.25], gap="medium")

    with left_col:
        with st.container(border=True):
            render_small_panel_title("Categorías del modelo", "◈")

            cat_rows = ""
            for key, label in CATEGORY_LABELS.items():
                cat_rows += (
                    f"<tr>"
                    f"<td style='padding:.55rem .75rem;border-bottom:1px solid #f0f4fa;"
                    f"color:var(--muted);font-size:.84rem;font-weight:700'>{escape(key)}</td>"
                    f"<td style='padding:.55rem .75rem;border-bottom:1px solid #f0f4fa'>"
                    f"{_cat_badge(key, label)}</td>"
                    f"</tr>"
                )

            st.markdown(
                '<table style="width:100%;border-collapse:collapse">'
                '<thead><tr>'
                '<th style="background:#f8fafc;color:#526079;font-size:.75rem;font-weight:800;'
                'padding:.6rem .75rem;text-align:left;border-bottom:1px solid var(--panel-border)">'
                'Categoría técnica</th>'
                '<th style="background:#f8fafc;color:#526079;font-size:.75rem;font-weight:800;'
                'padding:.6rem .75rem;text-align:left;border-bottom:1px solid var(--panel-border)">'
                'Nombre visible</th>'
                f'</tr></thead><tbody>{cat_rows}</tbody></table>',
                unsafe_allow_html=True,
            )

        st.markdown("<div style='height:.65rem'></div>", unsafe_allow_html=True)

        with st.container(border=True):
            render_small_panel_title("Métricas de confianza", "◎")
            st.markdown(
                '<div class="mdl-confidence-copy">'
                'Las métricas de confianza indican qué tan seguro está el modelo al '
                'clasificar un producto. El sistema clasifica las predicciones en tres '
                'niveles: alta confianza cuando el resultado es mayor o igual al 80%, '
                'confianza media entre 50% y 79.99%, y baja confianza cuando es menor '
                'al 50%. Estas métricas ayudan a decidir si un producto puede '
                'registrarse directamente o si debe revisarse manualmente.'
                '</div>',
                unsafe_allow_html=True,
            )

    with right_col:
        with st.container(border=True):
            render_small_panel_title("Resumen técnico", "⚙")

            details = [
                ("◎", "Base del entrenamiento", "Transfer Learning"),
                ("▣", "Backbone",               "MobileNetV2"),
                ("◎", "Objetivo",               "Clasificación de productos de despensa"),
                ("▤", "Salida",                 "Categoría predicha y confianza"),
            ]
            detail_rows = "".join(
                f'<div class="mdl-detail-row">'
                f'<span class="mdl-detail-icon">{icon}</span>'
                f'<span class="mdl-detail-label">{escape(lbl)}</span>'
                f'<span class="mdl-detail-value">{escape(val)}</span>'
                f'</div>'
                for icon, lbl, val in details
            )
            st.markdown(
                f'<div style="background:#fff;border:1px solid var(--panel-border);'
                f'border-radius:8px;padding:.75rem 1rem">{detail_rows}</div>',
                unsafe_allow_html=True,
            )

            with st.expander("Ver detalles del modelo →"):
                if report_ok:
                    try:
                        report = json.loads(report_path.read_text(encoding="utf-8"))
                        st.json(report)
                    except Exception:
                        st.caption("No se pudo leer el reporte.")
                else:
                    st.caption("No se encontró training_report.json.")

        st.markdown("<div style='height:.65rem'></div>", unsafe_allow_html=True)

        model_files = [
            "trained_models/product_classifier.keras",
            "trained_models/training_report.json",
            "trained_models/accuracy_loss.png",
            "trained_models/confusion_matrix.png",
        ]

        with st.container(border=True):
            render_small_panel_title("Archivos principales del modelo", "▤")

            file_rows = "".join(
                f'<div class="mdl-file-row">'
                f'<span class="mdl-file-icon">▢</span>'
                f'<span class="mdl-file-path">{escape(fp)}</span>'
                f'</div>'
                for fp in model_files
            )
            st.markdown(
                f'<div style="background:#fff;border:1px solid var(--panel-border);'
                f'border-radius:8px;padding:.75rem 1rem;margin-bottom:.5rem">{file_rows}</div>',
                unsafe_allow_html=True,
            )
            st.code("\n".join(model_files), language=None)

    # ── Bottom banner ──────────────────────────────────────────────────────
    if report_ok:
        b_title = "Reporte de entrenamiento encontrado."
        b_copy  = ("Gráficas y métricas de entrenamiento disponibles para consulta "
                   "en la sección de detalles del modelo.")
    else:
        b_title = "Reporte de entrenamiento no encontrado."
        b_copy  = ("Ejecuta el entrenamiento del modelo para generar el reporte "
                   "con métricas y gráficas.")

    st.markdown(
        f'<div class="inv-banner">'
        f'<div>'
        f'<div class="inv-banner-title">{escape(b_title)}</div>'
        f'<div class="inv-banner-copy">{escape(b_copy)}</div>'
        f'</div>'
        f'<div class="inv-banner-icon">▲▤</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(
        page_title="Inventario Inteligente",
        page_icon="▣",
        layout="wide",
    )

    apply_custom_styles()
    render_app_header()

    api_online = check_api_status()

    render_status_banner(api_online)

    if not api_online:
        st.stop()

    if not is_logged_in():
        auth_view = st.session_state.get("auth_view", "login")

        if auth_view == "register":
            register_mode = st.session_state.get("register_mode", "bootstrap")
            show_register_page(mode=register_mode)
        else:
            show_login_page()

        st.stop()

    render_authenticated_sidebar()

    tab_names = [
        "Clasificar producto",
        "Cargue masivo",
        "Inventario",
        "Logs",
    ]
    if current_user_role() == "admin":
        tab_names.append("Usuarios")
        tab_names.append("Resultados del modelo")

    
    tab_names.append("Acerca del modelo")

    tabs = st.tabs(tab_names)

    with tabs[0]:
        if "prediction_response" not in st.session_state:
            st.session_state.prediction_response = None

        left_col, right_col = st.columns([1.55, 1], gap="medium")
        uploaded_file = None

        with left_col:
            with st.container(border=True):
                render_panel_title("Clasificar y registrar producto", "▧")
                render_field_label("Selecciona el metodo de entrada")

                input_method = st.radio(
                    "Selecciona el metodo de entrada",
                    options=["Subir imagen", "Usar camara"],
                    horizontal=True,
                    label_visibility="collapsed",
                )

                render_field_label("Sube una imagen del producto")

                if input_method == "Subir imagen":
                    uploaded_file = st.file_uploader(
                        "Sube una imagen del producto",
                        type=["jpg", "jpeg", "png", "webp"],
                        label_visibility="collapsed",
                    )
                else:
                    uploaded_file = st.camera_input(
                        "Toma una foto del producto",
                        label_visibility="collapsed",
                    )

                if uploaded_file is not None:
                    file_key = f"{uploaded_file.name}:{getattr(uploaded_file, 'size', 0)}"
                    if st.session_state.get("uploaded_image_key") != file_key:
                        st.session_state.uploaded_image_key = file_key
                        st.session_state.prediction_response = None

                    if st.button(
                        "Clasificar producto",
                        type="primary",
                        use_container_width=True,
                    ):
                        with st.spinner("Clasificando imagen..."):
                            prediction_response = predict_image(uploaded_file)

                        if prediction_response:
                            st.session_state.prediction_response = prediction_response
                            st.session_state.uploaded_image_name = uploaded_file.name

                prediction_response = st.session_state.get("prediction_response")

                if prediction_response and uploaded_file is not None:
                    prediction = prediction_response["prediction"]
                    is_classifiable = prediction.get("is_classifiable", True)

                    if not is_classifiable:
                        st.warning(
                            "La predicción está por debajo del umbral de registro directo. "
                            "Revisa el nivel de confianza, corrige manualmente si aplica "
                            "o intenta con otra imagen más clara."
                        )
                    else:
                        generated_code = prediction_response["generated_code"]

                        st.markdown('<div class="confirm-grid"></div>', unsafe_allow_html=True)
                        render_small_panel_title("Confirmar registro", "✓")

                        suggested_name = (
                            Path(uploaded_file.name).stem
                            if uploaded_file.name
                            else "producto_capturado"
                        )

                        product_name = st.text_input(
                            "Nombre del producto",
                            value=suggested_name,
                        )

                        quantity_col, category_col = st.columns([0.7, 1.3])

                        with quantity_col:
                            quantity = st.number_input(
                                "Cantidad",
                                min_value=1,
                                value=1,
                                step=1,
                            )

                        predicted_category = prediction["predicted_category"]
                        category_options = list(CATEGORY_LABELS.keys())

                        with category_col:
                            selected_category = st.selectbox(
                                "Categoria",
                                options=category_options,
                                index=category_options.index(predicted_category)
                                if predicted_category in category_options
                                else 0,
                                format_func=format_category,
                            )

                        confidence = float(prediction["confidence"])

                        if st.button(
                            "Guardar en inventario",
                            type="primary",
                            use_container_width=True,
                        ):
                            # 1. Consultamos los productos que ya existen
                            current_inventory = get_products()
                            # 2. Extraemos los nombres en minúsculas para comparar sin importar mayúsculas
                            existing_names = [str(p.get("name", "")).strip().lower() for p in current_inventory]
                            
                            # 3. Verificamos si el nombre que el usuario quiere guardar ya existe
                            if product_name.strip().lower() in existing_names:
                                st.error(f"⚠️ Ya existe un producto registrado con el nombre '{product_name}'. Por favor, cámbialo antes de guardar.")
                            else:
                                # Si no existe, procedemos a guardar normalmente
                                payload = {
                                    "code": generated_code,
                                    "name": product_name,
                                    "category": selected_category,
                                    "quantity": int(quantity),
                                    "image_path": uploaded_file.name or "captura_camara.jpg",
                                    "confidence": confidence,
                                    "audit_details": "Registro desde clasificacion individual.",
                                }

                                result = register_product(payload)

                                if result:
                                    product = result.get("product", {})

                                    st.success("Producto registrado correctamente.")

                                    st.markdown(
                                        f"""
                                        **Producto:** {product.get("name", "-")}  
                                        **Código:** {product.get("code", "-")}  
                                        **Categoría:** {format_category(product.get("category", "-"))}  
                                        **Cantidad:** {product.get("quantity", "-")}
                                        """
                                    )

                                    st.session_state.prediction_response = None

        with right_col:
            with st.container(border=True):
                render_small_panel_title("Resultado de la clasificacion", "✦")

                preview_col, result_col = st.columns([0.9, 1.25], gap="small")

                with preview_col:
                    if uploaded_file is not None:
                        st.image(
                            uploaded_file,
                            caption="Vista previa del producto",
                            use_container_width=True,
                        )
                    else:
                        render_empty_preview()

                with result_col:
                    _pred_resp = st.session_state.get("prediction_response")
                    _visible_pred = _pred_resp
                    render_result_cards(_visible_pred, format_category)
                    if _visible_pred is not None:
                        render_confidence_alert(
                            confidence_percent_from_result(_visible_pred["prediction"])
                        )

                render_result_details(_visible_pred, format_category)

        if uploaded_file is None:
            render_info_strip(
                "Sube una imagen para iniciar la clasificacion.",
                "El sistema analizara tu producto y sugerira la categoria mas probable.",
            )
        elif st.session_state.get("prediction_response") is None:
            render_info_strip(
                "Imagen lista para clasificar.",
                "Presiona Clasificar producto para consultar el modelo ONNX.",
            )

    with tabs[1]:
        if "batch_upload_version" not in st.session_state:
            st.session_state.batch_upload_version = 0
        if "batch_signature" not in st.session_state:
            st.session_state.batch_signature = ""

        left_col, right_col = st.columns([1.55, 1], gap="medium")
        uploaded_files = None
        uploaded_zip = None

        with left_col:
            with st.container(border=True):
                render_panel_title("Cargue masivo de imagenes", "⇧")
                render_field_label("Selecciona el modo de entrada")

                input_mode = st.radio(
                    "Selecciona el modo de entrada",
                    options=["Subir imagenes", "Subir ZIP"],
                    horizontal=True,
                    key="batch_input_mode",
                    label_visibility="collapsed",
                )

                render_field_label(
                    "Sube una o mas imagenes del producto"
                    if input_mode == "Subir imagenes"
                    else "Sube un archivo ZIP con imagenes"
                )

                upload_version = st.session_state.batch_upload_version

                if input_mode == "Subir imagenes":
                    uploaded_files = st.file_uploader(
                        "Sube una o mas imagenes del producto",
                        type=["jpg", "jpeg", "png", "webp"],
                        accept_multiple_files=True,
                        key=f"batch_files_{upload_version}",
                        label_visibility="collapsed",
                    )
                else:
                    uploaded_zip = st.file_uploader(
                        "Sube un archivo ZIP con imagenes",
                        type=["zip"],
                        key=f"batch_zip_{upload_version}",
                        label_visibility="collapsed",
                    )

                file_signature = "|".join(
                    f"{file.name}:{getattr(file, 'size', 0)}"
                    for file in (uploaded_files or [])
                )
                if uploaded_zip is not None:
                    file_signature = f"{uploaded_zip.name}:{getattr(uploaded_zip, 'size', 0)}"

                if file_signature != st.session_state.batch_signature:
                    st.session_state.batch_signature = file_signature
                    st.session_state["batch_result"] = None

                file_list_placeholder = st.empty()

                can_classify = bool(
                    (input_mode == "Subir imagenes" and uploaded_files)
                    or (input_mode == "Subir ZIP" and uploaded_zip)
                )

                action_col, clear_col = st.columns([0.85, 1.15])

                with action_col:
                    classify_clicked = st.button(
                        "Clasificar todo",
                        disabled=not can_classify,
                        key="btn_batch_classify",
                        type="primary",
                        use_container_width=True,
                    )

                with clear_col:
                    clear_clicked = st.button(
                        "Limpiar lista",
                        disabled=not can_classify and not st.session_state.get("batch_result"),
                        key="btn_batch_clear",
                        use_container_width=True,
                    )

                if clear_clicked:
                    st.session_state.batch_upload_version += 1
                    st.session_state.batch_signature = ""
                    st.session_state["batch_result"] = None
                    st.rerun()

                if classify_clicked:
                    with st.spinner("Clasificando imagenes..."):
                        batch_result = predict_batch_files(
                            files_data=uploaded_files
                            if input_mode == "Subir imagenes"
                            else None,
                            zip_data=uploaded_zip if input_mode == "Subir ZIP" else None,
                        )
                    if batch_result is not None:
                        st.session_state["batch_result"] = batch_result

                with file_list_placeholder:
                    render_batch_file_list(
                        uploaded_files,
                        uploaded_zip,
                        st.session_state.get("batch_result"),
                    )

                batch = st.session_state.get("batch_result")

                if batch:
                    results = batch.get("results", [])
                    ok_results = [r for r in results if "error" not in r]
                    error_results = [r for r in results if "error" in r]

                    if ok_results:
                        classifiable = [r for r in ok_results if r.get("is_classifiable", True)]
                        not_classif  = [r for r in ok_results if not r.get("is_classifiable", True)]
                        confidence_metrics = calculate_confidence_metrics(
                            [confidence_percent_from_result(r) for r in ok_results]
                        )

                        render_small_panel_title("Resultados de clasificacion", "✓")
                        render_confidence_metric_cards(confidence_metrics)

                        # --- CONTROL DE ELEMENTOS ELIMINADOS ---
                        if "deleted_batch_items" not in st.session_state or st.session_state.get("last_batch_signature") != st.session_state.batch_signature:
                            st.session_state.deleted_batch_items = set()
                            st.session_state.last_batch_signature = st.session_state.batch_signature

                        # Filtramos las imágenes eliminadas
                        visible_raw_results = [r for r in ok_results if r["filename"] not in st.session_state.deleted_batch_items]

                        if not visible_raw_results:
                            st.info("No hay imágenes en la lista. Todas han sido eliminadas.")
                        else:
                            st.markdown(
                                "<div style='margin-top: 1rem; margin-bottom: 0.5rem; font-size: 0.95rem; color: var(--muted);'>"
                                "💡 <b>Tip:</b> Edita Nombre, Cantidad o Categoría. Desmarca 'Guardar' para omitir, o presiona 🗑️ para quitar de la lista por completo."
                                "</div>",
                                unsafe_allow_html=True
                            )

                            # --- ENCABEZADOS DE LA LISTA CON TODAS TUS COLUMNAS ---
                            h1, h2, h3, h4, h5, h6, h7, h8 = st.columns([0.6, 2, 0.9, 1.4, 1.8, 0.9, 1.1, 0.6])
                            h1.markdown("<div style='font-size:0.8rem;font-weight:bold;color:#526079;'>Guardar</div>", unsafe_allow_html=True)
                            h2.markdown("<div style='font-size:0.8rem;font-weight:bold;color:#526079;'>Nombre</div>", unsafe_allow_html=True)
                            h3.markdown("<div style='font-size:0.8rem;font-weight:bold;color:#526079;'>Cant.</div>", unsafe_allow_html=True)
                            h4.markdown("<div style='font-size:0.8rem;font-weight:bold;color:#526079;'>Categoría</div>", unsafe_allow_html=True)
                            h5.markdown("<div style='font-size:0.8rem;font-weight:bold;color:#526079;'>Archivo</div>", unsafe_allow_html=True)
                            h6.markdown("<div style='font-size:0.8rem;font-weight:bold;color:#526079;'>Confianza</div>", unsafe_allow_html=True)
                            h7.markdown("<div style='font-size:0.8rem;font-weight:bold;color:#526079;'>Estado</div>", unsafe_allow_html=True)
                            h8.markdown("<div style='font-size:0.8rem;font-weight:bold;color:#526079;'>Eliminar</div>", unsafe_allow_html=True)

                            st.markdown("<hr style='margin: 0.2rem 0 0.5rem 0; border-color: #f0f4fa;'>", unsafe_allow_html=True)

                            formatted_categories = [format_category(c) for c in CATEGORY_LABELS.keys()]
                            to_save_data = []

                            # --- DIBUJAR CADA FILA (SIN OCULTAR NADA) ---
                            for r in visible_raw_results:
                                fname = r["filename"]
                                is_classif = r.get("is_classifiable", True)
                                estado_txt = "✓ Clasificado" if is_classif else "⚠ No clasificable"
                                
                                col1, col2, col3, col4, col5, col6, col7, col8 = st.columns([0.6, 2, 0.9, 1.4, 1.8, 0.9, 1.1, 0.6])
                                
                                with col1:
                                    st.markdown("<div style='height: 0.35rem;'></div>", unsafe_allow_html=True)
                                    guardar = st.checkbox("Guardar", value=is_classif, key=f"keep_{fname}", label_visibility="collapsed")
                                with col2:
                                    edited_name = st.text_input("Nombre", value=Path(fname).stem, key=f"name_{fname}", label_visibility="collapsed")
                                with col3:
                                    edited_qty = st.number_input("Cant", min_value=1, value=1, step=1, key=f"qty_{fname}", label_visibility="collapsed")
                                with col4:
                                    current_cat = format_category(r["predicted_category"])
                                    cat_idx = formatted_categories.index(current_cat) if current_cat in formatted_categories else 0
                                    edited_cat = st.selectbox("Cat", options=formatted_categories, index=cat_idx, key=f"cat_{fname}", label_visibility="collapsed")
                                with col5:
                                    # Muestra el nombre del archivo cortado elegantemente si es muy largo
                                    st.markdown(f"<div style='padding-top: 0.6rem; font-size: 0.8rem; color: var(--muted); overflow: hidden; text-overflow: ellipsis; white-space: nowrap;' title='{fname}'>{fname}</div>", unsafe_allow_html=True)
                                with col6:
                                    st.markdown(f"<div style='padding-top: 0.6rem; font-size: 0.8rem; color: var(--muted);'>{confidence_percent_from_result(r):.2f}%</div>", unsafe_allow_html=True)
                                with col7:
                                    st.markdown(f"<div style='padding-top: 0.6rem; font-size: 0.8rem; color: var(--muted);'>{estado_txt}</div>", unsafe_allow_html=True)
                                with col8:
                                    if st.button("🗑️", key=f"del_{fname}"):
                                        st.session_state.deleted_batch_items.add(fname)
                                        st.rerun()

                                # Guardamos toda la configuración elegida por el usuario
                                to_save_data.append({
                                    "Guardar": guardar,
                                    "Nombre": edited_name,
                                    "Cantidad": edited_qty,
                                    "Categoría": edited_cat,
                                    "Archivo": fname,
                                    "_code": r["generated_code"],
                                    "_tech_category": r["predicted_category"],
                                    "_confidence": r["confidence"]
                                })

                            export_col, save_col = st.columns(2)

                            with export_col:
                                _batch_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                                st.download_button(
                                    "⬇ Exportar Excel",
                                    data=_build_batch_xlsx(visible_raw_results), 
                                    file_name=f"clasificacion_masiva_{_batch_ts}.xlsx",
                                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                    key="btn_export_xlsx",
                                    use_container_width=True,
                                )

                            with save_col:
                                save_clicked = st.button(
                                    "Guardar seleccionados",
                                    key="btn_batch_save",
                                    type="primary",
                                    use_container_width=True,
                                )

                            if save_clicked:
                                # Solo procesamos las filas que dejaron con el 'check' de Guardar activado
                                to_save_filtered = [row for row in to_save_data if row["Guardar"]]
                                
                                if not to_save_filtered:
                                    st.warning("No seleccionaste ninguna imagen para guardar (casillas desmarcadas).")
                                else:
                                    progress = st.progress(0)
                                    saved = 0
                                    failed = 0
                                    total = len(to_save_filtered)
                                    category_reverse_map = {v: k for k, v in CATEGORY_LABELS.items()}

                                    # --- VALIDACIÓN DE NOMBRES DUPLICADOS ---
                                    current_inventory = get_products()
                                    existing_names = [str(p.get("name", "")).strip().lower() for p in current_inventory]
                                    
                                    has_duplicates = False
                                    for row in to_save_filtered:
                                        if str(row["Nombre"]).strip().lower() in existing_names:
                                            has_duplicates = True
                                            st.error(f"⚠️ El producto '{row['Nombre']}' ya existe en el inventario. Cambia su nombre antes de guardar.")
                                    
                                    if not has_duplicates:
                                        for i, row in enumerate(to_save_filtered):
                                            tech_category = category_reverse_map.get(row["Categoría"], row["_tech_category"])
                                            payload = {
                                                "code": row["_code"],
                                                "name": str(row["Nombre"]).strip(),
                                                "category": tech_category,
                                                "quantity": int(row["Cantidad"]),
                                                "image_path": str(row["Archivo"]),
                                                "confidence": float(row["_confidence"]),
                                                "audit_details": "Registro desde cargue masivo (editado en lista personalizada).",
                                            }
                                            result_response = register_product(payload, show_errors=False)
                                            if result_response:
                                                saved += 1
                                            else:
                                                failed += 1
                                            progress.progress((i + 1) / total)

                                        summary = f"Guardados: {saved}"
                                        st.success(summary)
                                        if failed > 0:
                                            st.warning(f"{failed} producto(s) no se pudieron guardar (código duplicado en BD).")

                    if error_results:
                        with st.expander(f"Imagenes con error ({len(error_results)})"):
                            for result in error_results:
                                st.error(f"**{result['filename']}** - {result['error']}")

        with right_col:
            with st.container(border=True):
                render_small_panel_title("Resumen del lote", "▤")
                render_batch_summary(
                    uploaded_files,
                    uploaded_zip,
                    st.session_state.get("batch_result"),
                    format_category,
                    [format_category(category) for category in CATEGORY_LABELS],
                )
                render_batch_details(
                    st.session_state.get("batch_result"),
                    format_category,
                )

        if not st.session_state.get("batch_result"):
            render_info_strip(
                "Sube imagenes o un ZIP y presiona 'Clasificar todo' para ver los resultados.",
                "El sistema procesara el lote y mostrara categorias, confianza y registros creados.",
            )
        else:
            render_info_strip(
                "Lote procesado correctamente.",
                "Revisa los resultados, exporta el CSV o guarda los productos en inventario.",
            )

    with tabs[2]:
        render_inventory_tab()

    with tabs[3]:
        render_logs_tab()

    next_tab_index = 4
    # Aquí metemos ambas pestañas restringidas dentro del mismo 'if'
    if current_user_role() == "admin":
        with tabs[next_tab_index]:
            render_users_tab()
        next_tab_index += 1
        
        with tabs[next_tab_index]:
            render_model_results_tab()
        next_tab_index += 1

    # La última pestaña ("Acerca del modelo") se renderiza para todos
    with tabs[next_tab_index]:
        render_model_tab()

if __name__ == "__main__":
    main()
