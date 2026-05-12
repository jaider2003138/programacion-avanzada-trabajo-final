"""
Prueba individual del modelo clasificador de productos de despensa.

Ejemplo de uso desde la raíz del proyecto:

    python scripts/predict_image.py --image datasets/processed/test/cafe_chocolate/imagen.jpg

También puedes pasar cualquier imagen externa:

    python scripts/predict_image.py --image ruta/a/mi/imagen.jpg
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps


MODEL_PATH = Path("trained_models/product_classifier.keras")
LABELS_PATH = Path("datasets/processed/labels.json")

IMAGE_SIZE = (224, 224)
TOP_K = 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clasificar una imagen individual usando el modelo entrenado."
    )

    parser.add_argument(
        "--image",
        required=True,
        help="Ruta de la imagen que se desea clasificar.",
    )

    parser.add_argument(
        "--model",
        default=str(MODEL_PATH),
        help="Ruta del modelo entrenado.",
    )

    parser.add_argument(
        "--labels",
        default=str(LABELS_PATH),
        help="Ruta del archivo labels.json.",
    )

    return parser.parse_args()


def load_labels(labels_path: Path) -> list[str]:
    if not labels_path.exists():
        raise FileNotFoundError(f"No se encontró el archivo de etiquetas: {labels_path}")

    with labels_path.open("r", encoding="utf-8") as file:
        labels_dict = json.load(file)

    class_names = [
        labels_dict[str(index)]
        for index in range(len(labels_dict))
    ]

    return class_names


def load_and_preprocess_image(image_path: Path) -> np.ndarray:
    if not image_path.exists():
        raise FileNotFoundError(f"No se encontró la imagen: {image_path}")

    image = Image.open(image_path)
    image = ImageOps.exif_transpose(image)
    image = image.convert("RGB")
    image = image.resize(IMAGE_SIZE)

    image_array = np.array(image, dtype=np.float32)

    # El modelo ya contiene preprocess_input dentro de su arquitectura,
    # por eso aquí NO normalizamos manualmente a [-1, 1].
    image_array = np.expand_dims(image_array, axis=0)

    return image_array


def predict_image(
    model: tf.keras.Model,
    image_array: np.ndarray,
    class_names: list[str],
) -> dict:
    predictions = model.predict(image_array, verbose=0)[0]

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

    result = {
        "predicted_category": class_names[best_index],
        "confidence": float(predictions[best_index]),
        "confidence_percent": float(predictions[best_index] * 100),
        "top_predictions": top_predictions,
    }

    return result


def print_result(image_path: Path, result: dict) -> None:
    print("\nResultado de predicción")
    print("=" * 40)
    print(f"Imagen: {image_path}")
    print(f"Categoría predicha: {result['predicted_category']}")
    print(f"Confianza: {result['confidence_percent']:.2f}%")

    print("\nTop predicciones:")
    for index, item in enumerate(result["top_predictions"], start=1):
        print(
            f"{index}. {item['category']} - "
            f"{item['confidence_percent']:.2f}%"
        )

    print("=" * 40)


def main() -> None:
    args = parse_args()

    image_path = Path(args.image)
    model_path = Path(args.model)
    labels_path = Path(args.labels)

    if not model_path.exists():
        raise FileNotFoundError(f"No se encontró el modelo entrenado: {model_path}")

    print("\nCargando modelo...")
    model = tf.keras.models.load_model(model_path)

    print("Cargando etiquetas...")
    class_names = load_labels(labels_path)

    print("Procesando imagen...")
    image_array = load_and_preprocess_image(image_path)

    print("Realizando predicción...")
    result = predict_image(model, image_array, class_names)

    print_result(image_path, result)


if __name__ == "__main__":
    main()