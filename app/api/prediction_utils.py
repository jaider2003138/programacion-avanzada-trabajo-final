"""
Utilidades para cargar el modelo y clasificar imágenes desde la API Flask.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageOps


MODEL_PATH = Path("trained_models/product_classifier.onnx")
LABELS_PATH = Path("datasets/processed/labels.json")

IMAGE_SIZE = (224, 224)
TOP_K = 3
CONFIDENCE_THRESHOLD = 0.70


_session: ort.InferenceSession | None = None
_class_names: list[str] | None = None

HIGH_CONFIDENCE_LEVEL = "Alta confianza"
MEDIUM_CONFIDENCE_LEVEL = "Confianza media"
LOW_CONFIDENCE_LEVEL = "Baja confianza"


def normalize_confidence_percent(confidence: float | int | str) -> float:
    """
    Normaliza una confianza expresada como 0-1 o 0-100 a porcentaje.
    """
    confidence_value = float(confidence)

    if confidence_value <= 1:
        confidence_value *= 100

    return min(max(confidence_value, 0.0), 100.0)


def get_confidence_level(confidence_percent: float) -> str:
    """
    Clasifica la confianza del modelo en alta, media o baja.
    """
    normalized_confidence = normalize_confidence_percent(confidence_percent)

    if normalized_confidence >= 80:
        return HIGH_CONFIDENCE_LEVEL
    if normalized_confidence >= 50:
        return MEDIUM_CONFIDENCE_LEVEL
    return LOW_CONFIDENCE_LEVEL


def calculate_confidence_metrics(confidences: list[float]) -> dict[str, float | int]:
    """
    Calcula metricas agregadas de confianza para valores 0-1 o 0-100.
    """
    normalized_confidences: list[float] = []

    for confidence in confidences:
        try:
            normalized_confidence = normalize_confidence_percent(confidence)
        except (TypeError, ValueError):
            continue

        if np.isfinite(normalized_confidence):
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


def load_labels(labels_path: Path = LABELS_PATH) -> list[str]:
    """
    Carga las etiquetas del archivo labels.json.
    """
    if not labels_path.exists():
        raise FileNotFoundError(f"No se encontró labels.json: {labels_path}")

    with labels_path.open("r", encoding="utf-8") as file:
        labels_dict = json.load(file)

    return [labels_dict[str(index)] for index in range(len(labels_dict))]


def get_model() -> ort.InferenceSession:
    """
    Carga la sesión ONNX una sola vez y la reutiliza.
    """
    global _session

    if _session is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"No se encontró el modelo: {MODEL_PATH}")

        _session = ort.InferenceSession(str(MODEL_PATH))

    return _session


def get_class_names() -> list[str]:
    """
    Carga las clases una sola vez y las reutiliza.
    """
    global _class_names

    if _class_names is None:
        _class_names = load_labels()

    return _class_names


def preprocess_image(image: Image.Image) -> np.ndarray:
    """
    Prepara una imagen PIL para el modelo.
    """
    image = ImageOps.exif_transpose(image)
    image = image.convert("RGB")
    image = image.resize(IMAGE_SIZE)

    image_array = np.array(image, dtype=np.float32)
    image_array = np.expand_dims(image_array, axis=0)

    return image_array


def predict_pil_image(image: Image.Image) -> dict[str, Any]:
    """
    Realiza predicción sobre una imagen PIL.
    """
    session = get_model()
    class_names = get_class_names()

    image_array = preprocess_image(image)

    input_name = session.get_inputs()[0].name
    predictions = session.run(None, {input_name: image_array})[0][0]
    top_indices = predictions.argsort()[-TOP_K:][::-1]

    top_predictions = []

    for index in top_indices:
        top_predictions.append(
            {
                "category": class_names[index],
                "confidence": float(predictions[index]),
                "confidence_percent": float(predictions[index] * 100),
            }
        )

    best_index = int(top_indices[0])
    best_confidence = float(predictions[best_index])
    best_confidence_percent = best_confidence * 100
    is_classifiable = best_confidence >= CONFIDENCE_THRESHOLD

    return {
        "predicted_category": class_names[best_index],
        "confidence": best_confidence,
        "confidence_percent": best_confidence_percent,
        "confidence_level": get_confidence_level(best_confidence_percent),
        "top_predictions": top_predictions,
        "is_classifiable": is_classifiable,
        "classification_warning": (
            None if is_classifiable else
            "No fue posible clasificar esta imagen con suficiente confianza. "
            "Asegúrate de que la imagen muestre claramente un producto de "
            "despensa (aceite, arroz, pasta, enlatado, etc.)"
        ),
    }
