"""
Predice la categoría de una imagen y registra el producto en inventario.

Ejemplo de uso:

python scripts/predict_and_register.py --image "ruta/a/imagen.jpg" --name "Producto de prueba" --quantity 1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps


# Permite importar módulos desde la raíz del proyecto.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from app.services.code_generator import generate_inventory_code
from app.services.inventory_service import (
    get_next_sequence,
    initialize_database,
    register_product,
)


MODEL_PATH = Path("trained_models/product_classifier.keras")
LABELS_PATH = Path("datasets/processed/labels.json")

IMAGE_SIZE = (224, 224)
TOP_K = 3


def get_confidence_level(confidence_percent: float) -> str:
    """
    Clasifica la confianza del modelo segun el porcentaje recibido.
    """
    if confidence_percent >= 80:
        return "Alta confianza"
    if confidence_percent >= 50:
        return "Confianza media"
    return "Baja confianza"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clasificar una imagen y registrar el producto en inventario."
    )

    parser.add_argument(
        "--image",
        required=True,
        help="Ruta de la imagen del producto.",
    )

    parser.add_argument(
        "--name",
        default=None,
        help="Nombre del producto. Si no se indica, se usa el nombre del archivo.",
    )

    parser.add_argument(
        "--quantity",
        type=int,
        default=1,
        help="Cantidad inicial del producto.",
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

    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.50,
        help="Confianza mínima para registrar automáticamente.",
    )

    return parser.parse_args()


def load_labels(labels_path: Path) -> list[str]:
    if not labels_path.exists():
        raise FileNotFoundError(f"No se encontró labels.json: {labels_path}")

    with labels_path.open("r", encoding="utf-8") as file:
        labels_dict = json.load(file)

    return [labels_dict[str(index)] for index in range(len(labels_dict))]


def load_and_preprocess_image(image_path: Path) -> np.ndarray:
    if not image_path.exists():
        raise FileNotFoundError(f"No se encontró la imagen: {image_path}")

    image = Image.open(image_path)
    image = ImageOps.exif_transpose(image)
    image = image.convert("RGB")
    image = image.resize(IMAGE_SIZE)

    image_array = np.array(image, dtype=np.float32)

    # El modelo ya tiene preprocess_input internamente.
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

    return {
        "predicted_category": class_names[best_index],
        "confidence": float(predictions[best_index]),
        "confidence_percent": float(predictions[best_index] * 100),
        "top_predictions": top_predictions,
    }


def print_prediction_result(result: dict) -> None:
    print("\nResultado de clasificación")
    print("=" * 40)
    print(f"Categoría predicha: {result['predicted_category']}")
    print(f"Confianza: {result['confidence_percent']:.2f}%")
    print(f"Nivel de confianza: {get_confidence_level(result['confidence_percent'])}")

    print("\nTop predicciones:")
    for index, item in enumerate(result["top_predictions"], start=1):
        print(f"{index}. {item['category']} - {item['confidence_percent']:.2f}%")

    print("=" * 40)


def main() -> None:
    args = parse_args()

    image_path = Path(args.image)
    model_path = Path(args.model)
    labels_path = Path(args.labels)

    product_name = args.name or image_path.stem
    quantity = args.quantity

    if quantity <= 0:
        raise ValueError("La cantidad debe ser mayor que 0.")

    if not model_path.exists():
        raise FileNotFoundError(f"No se encontró el modelo: {model_path}")

    print("\nInicializando base de datos...")
    initialize_database()

    print("Cargando modelo...")
    model = tf.keras.models.load_model(model_path)

    print("Cargando etiquetas...")
    class_names = load_labels(labels_path)

    print("Procesando imagen...")
    image_array = load_and_preprocess_image(image_path)

    print("Clasificando producto...")
    result = predict_image(model, image_array, class_names)

    print_prediction_result(result)

    confidence = result["confidence"]
    predicted_category = result["predicted_category"]

    if confidence < args.min_confidence:
        print("\nAdvertencia:")
        confidence_level = get_confidence_level(result["confidence_percent"])
        print(
            f"La confianza ({result['confidence_percent']:.2f}%, {confidence_level}) "
            "está por debajo del mínimo requerido. No se registrará automáticamente."
        )
        print("Puedes registrar manualmente este producto más adelante.")
        return

    sequence = get_next_sequence()
    code = generate_inventory_code(predicted_category, sequence)

    product = register_product(
        code=code,
        name=product_name,
        category=predicted_category,
        quantity=quantity,
        image_path=str(image_path),
        confidence=confidence,
    )

    print("\nProducto registrado en inventario")
    print("=" * 40)
    print(f"ID: {product['id']}")
    print(f"Código: {product['code']}")
    print(f"Nombre: {product['name']}")
    print(f"Categoría: {product['category']}")
    print(f"Cantidad: {product['quantity']}")
    print(f"Confianza: {product['confidence'] * 100:.2f}%")
    print(f"Nivel de confianza: {get_confidence_level(product['confidence'] * 100)}")
    print(f"Fecha: {product['created_at']}")
    print(f"Estado: {product['status']}")
    print("=" * 40)


if __name__ == "__main__":
    main()
