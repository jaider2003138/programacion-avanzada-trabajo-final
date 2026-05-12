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

from pathlib import Path
from typing import Any

import pandas as pd
import requests
import streamlit as st


API_BASE_URL = "http://127.0.0.1:5000"


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
        page_icon="📦",
        layout="wide",
    )

    st.title("📦 Sistema Inteligente de Inventario")
    st.caption("Clasificación e inventariado de productos de despensa mediante imágenes")

    api_online = check_api_status()

    if api_online:
        st.success("API Flask conectada correctamente.")
    else:
        st.error(
            "La API Flask no está activa. Ejecuta primero: "
            "`python -m app.api.app`"
        )
        st.stop()

    tabs = st.tabs(
        [
            "Clasificar producto",
            "Inventario",
            "Acerca del modelo",
        ]
    )

    with tabs[0]:
        st.header("Clasificar y registrar producto")

        input_method = st.radio(
            "Selecciona el método de entrada",
            options=["Subir imagen", "Usar cámara"],
            horizontal=True,
        )

        uploaded_file = None

        if input_method == "Subir imagen":
            uploaded_file = st.file_uploader(
                "Sube una imagen del producto",
                type=["jpg", "jpeg", "png", "webp"],
            )

        elif input_method == "Usar cámara":
            uploaded_file = st.camera_input(
                "Toma una foto del producto"
            )

        if uploaded_file is not None:
            st.image(
                uploaded_file,
                caption="Imagen cargada",
                width=300,
            )

            if "prediction_response" not in st.session_state:
                st.session_state.prediction_response = None

            if st.button("Clasificar producto"):
                with st.spinner("Clasificando imagen..."):
                    prediction_response = predict_image(uploaded_file)

                if prediction_response:
                    st.session_state.prediction_response = prediction_response
                    st.session_state.uploaded_image_name = uploaded_file.name

            if st.session_state.get("prediction_response"):
                prediction_response = st.session_state.prediction_response
                render_prediction_result(prediction_response)

                prediction = prediction_response["prediction"]
                generated_code = prediction_response["generated_code"]

                st.subheader("Confirmar registro")

                suggested_name = Path(uploaded_file.name).stem if uploaded_file.name else "producto_capturado"

                product_name = st.text_input(
                    "Nombre del producto",
                    value=suggested_name,
                )

                quantity = st.number_input(
                    "Cantidad",
                    min_value=1,
                    value=1,
                    step=1,
                )

                predicted_category = prediction["predicted_category"]

                category_options = list(CATEGORY_LABELS.keys())

                selected_category = st.selectbox(
                    "Categoría",
                    options=category_options,
                    index=category_options.index(predicted_category)
                    if predicted_category in category_options
                    else 0,
                    format_func=format_category,
                )

                confidence = float(prediction["confidence"])

                if st.button("Guardar en inventario"):
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

                        # Limpia la predicción para evitar registrar lo mismo por accidente.
                        st.session_state.prediction_response = None

        else:
            st.info("Sube una imagen para iniciar la clasificación.")

    with tabs[1]:
        render_inventory_table()

        if st.button("Actualizar inventario"):
            st.rerun()

    with tabs[2]:
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