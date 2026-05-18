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


_session: ort.InferenceSession | None = None
_class_names: list[str] | None = None


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

    return {
        "predicted_category": class_names[best_index],
        "confidence": float(predictions[best_index]),
        "confidence_percent": float(predictions[best_index] * 100),
        "top_predictions": top_predictions,
    }