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

import os
from pathlib import Path
from typing import Any

import pandas as pd
import requests
import streamlit as st

try:
    from streamlit_app.components import (
        render_app_header,
        render_batch_details,
        render_batch_file_list,
        render_batch_summary,
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
        render_app_header,
        render_batch_details,
        render_batch_file_list,
        render_batch_summary,
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


def register_product(payload: dict[str, Any]) -> dict[str, Any] | None:
    """
    Envía un producto al endpoint /products.
    """
    try:
        response = requests.post(
            f"{API_BASE_URL}/products",
            json=payload,
            timeout=30,
        )

        if response.status_code not in [200, 201]:
            st.error("Error al registrar el producto.")
            try:
                st.json(response.json())
            except Exception:
                st.write(response.text)
            return None

        return response.json()

    except requests.RequestException as exc:
        st.error(f"No se pudo registrar el producto: {exc}")
        return None


def get_products() -> list[dict[str, Any]]:
    """
    Consulta productos registrados.
    """
    try:
        response = requests.get(f"{API_BASE_URL}/products", timeout=10)

        if response.status_code != 200:
            return []

        data = response.json()
        return data.get("products", [])

    except requests.RequestException:
        return []


def format_category(category: str) -> str:
    """
    Convierte el nombre técnico de la categoría a un nombre legible.
    """
    return CATEGORY_LABELS.get(category, category)


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
                timeout=300,
            )
        else:
            response = requests.post(
                f"{API_BASE_URL}/predict/batch",
                files={"zip": (zip_data.name, zip_data.getvalue(), "application/zip")},
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
    confidence_percent = prediction["confidence_percent"]

    st.subheader("Resultado de clasificación")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Categoría", format_category(predicted_category))

    with col2:
        st.metric("Confianza", f"{confidence_percent:.2f}%")

    with col3:
        st.metric("Código generado", generated_code)

    if confidence_percent >= 80:
        st.success("Predicción confiable.")
    elif confidence_percent >= 50:
        st.warning("Predicción con confianza media. Se recomienda revisar antes de guardar.")
    else:
        st.error("Predicción con baja confianza. Se recomienda corregir manualmente.")

    st.write("Top predicciones:")

    top_predictions = prediction.get("top_predictions", [])

    top_data = [
        {
            "Categoría": format_category(item["category"]),
            "Confianza (%)": round(item["confidence_percent"], 2),
        }
        for item in top_predictions
    ]

    st.dataframe(pd.DataFrame(top_data), use_container_width=True)


def render_inventory_table() -> None:
    """
    Muestra tabla de inventario.
    """
    products = get_products()

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

    tabs = st.tabs(
        [
            "▧  Clasificar producto",
            "⇧  Cargue masivo",
            "▣  Inventario",
            "ⓘ  Acerca del modelo",
        ]
    )

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
                        payload = {
                            "code": generated_code,
                            "name": product_name,
                            "category": selected_category,
                            "quantity": int(quantity),
                            "image_path": uploaded_file.name or "captura_camara.jpg",
                            "confidence": confidence,
                        }

                        result = register_product(payload)

                        if result:
                            st.success("Producto registrado correctamente.")
                            st.json(result)
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
                    render_result_cards(
                        st.session_state.get("prediction_response"),
                        format_category,
                    )

                render_result_details(
                    st.session_state.get("prediction_response"),
                    format_category,
                )

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
                        render_small_panel_title("Resultados de clasificacion", "✓")

                        table_data = [
                            {
                                "Archivo": r["filename"],
                                "Categoria": format_category(r["predicted_category"]),
                                "Confianza (%)": round(r["confidence_percent"], 2),
                            }
                            for r in ok_results
                        ]
                        df_results = pd.DataFrame(table_data)
                        st.dataframe(df_results, use_container_width=True)

                        export_col, save_col = st.columns(2)

                        with export_col:
                            csv_bytes = df_results.to_csv(index=False).encode("utf-8")
                            st.download_button(
                                "Exportar resultados",
                                data=csv_bytes,
                                file_name="resultados_clasificacion.csv",
                                mime="text/csv",
                                key="btn_export_csv",
                                use_container_width=True,
                            )

                        with save_col:
                            save_clicked = st.button(
                                "Guardar todo",
                                key="btn_batch_save",
                                type="primary",
                                use_container_width=True,
                            )

                        if save_clicked:
                            progress = st.progress(0)
                            saved = 0
                            failed = 0
                            total = len(ok_results)

                            for i, result in enumerate(ok_results):
                                payload = {
                                    "code": result["generated_code"],
                                    "name": Path(result["filename"]).stem,
                                    "category": result["predicted_category"],
                                    "quantity": 1,
                                    "image_path": result["filename"],
                                    "confidence": result["confidence"],
                                }
                                try:
                                    resp = requests.post(
                                        f"{API_BASE_URL}/products",
                                        json=payload,
                                        timeout=30,
                                    )
                                    if resp.status_code in (200, 201):
                                        saved += 1
                                    else:
                                        failed += 1
                                except requests.RequestException:
                                    failed += 1

                                progress.progress((i + 1) / total)

                            if saved > 0:
                                st.success(f"{saved} producto(s) guardado(s) en inventario.")
                            if failed > 0:
                                st.warning(
                                    f"{failed} producto(s) no se pudieron guardar "
                                    "(puede que el codigo ya exista en la BD)."
                                )

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
        render_inventory_table()

        if st.button("Actualizar inventario"):
            st.rerun()

    with tabs[3]:
        st.header("Información del modelo")

        st.write(
            """
            El modelo fue entrenado usando Transfer Learning con MobileNetV2
            para clasificar productos de despensa en las siguientes categorías:
            """
        )

        categories_df = pd.DataFrame(
            [
                {
                    "Categoría técnica": key,
                    "Nombre visible": value,
                }
                for key, value in CATEGORY_LABELS.items()
            ]
        )

        st.dataframe(categories_df, use_container_width=True)

        st.write("Archivos principales del modelo:")

        st.code(
            """
trained_models/product_classifier.keras
trained_models/training_report.json
trained_models/accuracy_loss.png
trained_models/confusion_matrix.png
            """.strip()
        )

        report_path = Path("trained_models/training_report.json")

        if report_path.exists():
            st.success("Reporte de entrenamiento encontrado.")
        else:
            st.warning("No se encontró training_report.json.")


if __name__ == "__main__":
    main()
