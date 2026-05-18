"""
API Flask para el Sistema Inteligente de Clasificación e Inventariado
de Productos de Despensa.

Ejecutar desde la raíz del proyecto:

    python app/api/app.py
"""

from __future__ import annotations

import io
import os
import sys
import zipfile
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, request
from flask_cors import CORS
from PIL import Image


# Permite importar módulos desde la raíz del proyecto.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from app.api.prediction_utils import predict_pil_image
from app.services.code_generator import generate_inventory_code
from app.services.inventory_service import (
    find_product_by_code,
    get_next_sequence,
    initialize_database,
    list_products,
    register_product,
)


app = Flask(__name__)
CORS(app)


@app.route("/", methods=["GET"])
def home():
    """
    Endpoint de prueba.
    """
    return jsonify(
        {
            "message": "API de Clasificación e Inventario de Productos de Despensa",
            "status": "running",
            "endpoints": [
                "GET /health",
                "POST /predict",
                "POST /predict/batch",
                "POST /products",
                "GET /products",
                "GET /products/<code>",
            ],
        }
    )


@app.route("/health", methods=["GET"])
def health():
    """
    Verifica que la API esté funcionando.
    """
    return jsonify(
        {
            "status": "ok",
            "message": "API funcionando correctamente",
        }
    )


@app.route("/predict", methods=["POST"])
def predict():
    """
    Recibe una imagen y devuelve la predicción del modelo.

    Form-data esperado:
    - image: archivo de imagen
    """
    if "image" not in request.files:
        return jsonify({"error": "No se envió ningún archivo con el campo 'image'."}), 400

    image_file = request.files["image"]

    if image_file.filename == "":
        return jsonify({"error": "El archivo de imagen está vacío."}), 400

    try:
        image = Image.open(image_file.stream)
        prediction = predict_pil_image(image)

        sequence = get_next_sequence()
        generated_code = generate_inventory_code(
            prediction["predicted_category"],
            sequence,
        )

        response = {
            "prediction": prediction,
            "generated_code": generated_code,
        }

        return jsonify(response), 200

    except Exception as exc:
        return jsonify(
            {
                "error": "Error procesando la imagen.",
                "detail": str(exc),
            }
        ), 500


@app.route("/predict/batch", methods=["POST"])
def predict_batch():
    """
    Clasifica múltiples imágenes en una sola llamada.

    Form-data esperado (solo uno de los dos campos):
    - files: uno o más archivos de imagen (jpg, jpeg, png, webp)
    - zip:   un archivo .zip que contiene imágenes (se procesa de forma recursiva)

    Límite: 100 imágenes por request.
    """
    VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
    MAX_IMAGES = 100

    has_files = bool(request.files.getlist("files"))
    has_zip = "zip" in request.files and request.files["zip"].filename != ""

    if not has_files and not has_zip:
        return jsonify(
            {"error": "Debes enviar 'files' (imágenes) o 'zip' (archivo ZIP)."}
        ), 400

    if has_files and has_zip:
        return jsonify(
            {"error": "Envía solo 'files' o solo 'zip', no ambos a la vez."}
        ), 400

    images: list[tuple[str, bytes]] = []

    if has_files:
        for uploaded in request.files.getlist("files"):
            if not uploaded.filename:
                continue
            if Path(uploaded.filename).suffix.lower() in VALID_EXTENSIONS:
                images.append((uploaded.filename, uploaded.read()))

    else:
        zip_file = request.files["zip"]
        try:
            with zipfile.ZipFile(io.BytesIO(zip_file.read())) as zf:
                for entry in zf.infolist():
                    if entry.is_dir():
                        continue
                    if Path(entry.filename).suffix.lower() in VALID_EXTENSIONS:
                        images.append((Path(entry.filename).name, zf.read(entry.filename)))
        except zipfile.BadZipFile:
            return jsonify({"error": "El archivo ZIP no es válido o está corrupto."}), 400

    total_received = len(images)

    if total_received > MAX_IMAGES:
        return jsonify(
            {
                "error": (
                    f"Se detectaron {total_received} imágenes válidas. "
                    f"El máximo permitido es {MAX_IMAGES}."
                ),
                "total_received": total_received,
                "limit": MAX_IMAGES,
            }
        ), 400

    if total_received == 0:
        return jsonify(
            {"total_received": 0, "total_processed": 0, "total_skipped": 0, "results": []}
        ), 200

    base_sequence = get_next_sequence()
    results: list[dict[str, Any]] = []
    total_skipped = 0

    for offset, (filename, raw_bytes) in enumerate(images):
        try:
            image = Image.open(io.BytesIO(raw_bytes))
            prediction = predict_pil_image(image)
            generated_code = generate_inventory_code(
                prediction["predicted_category"],
                base_sequence + offset,
            )
            results.append(
                {
                    "filename": filename,
                    "predicted_category": prediction["predicted_category"],
                    "confidence": prediction["confidence"],
                    "confidence_percent": prediction["confidence_percent"],
                    "top_predictions": prediction["top_predictions"],
                    "generated_code": generated_code,
                }
            )
        except Exception as exc:
            total_skipped += 1
            results.append({"filename": filename, "error": str(exc)})

    return jsonify(
        {
            "total_received": total_received,
            "total_processed": total_received - total_skipped,
            "total_skipped": total_skipped,
            "results": results,
        }
    ), 200


@app.route("/products", methods=["POST"])
def create_product():
    """
    Registra un producto en inventario.

    JSON esperado:
    {
      "code": "INV-CAF-202605-0001",
      "name": "Nescafe Tradicion 170g",
      "category": "cafe_chocolate",
      "quantity": 5,
      "image_path": "ruta/a/imagen.jpg",
      "confidence": 0.99
    }
    """
    data: dict[str, Any] = request.get_json(silent=True) or {}

    required_fields = [
        "code",
        "name",
        "category",
        "quantity",
        "image_path",
        "confidence",
    ]

    missing_fields = [
        field for field in required_fields
        if field not in data
    ]

    if missing_fields:
        return jsonify(
            {
                "error": "Faltan campos obligatorios.",
                "missing_fields": missing_fields,
            }
        ), 400

    try:
        product = register_product(
            code=str(data["code"]),
            name=str(data["name"]),
            category=str(data["category"]),
            quantity=int(data["quantity"]),
            image_path=str(data["image_path"]),
            confidence=float(data["confidence"]),
        )

        return jsonify(
            {
                "message": "Producto registrado correctamente.",
                "product": product,
            }
        ), 201

    except Exception as exc:
        return jsonify(
            {
                "error": "Error registrando el producto.",
                "detail": str(exc),
            }
        ), 500


@app.route("/products", methods=["GET"])
def get_products():
    """
    Lista los productos del inventario.
    """
    try:
        products = list_products()

        return jsonify(
            {
                "total": len(products),
                "products": products,
            }
        ), 200

    except Exception as exc:
        return jsonify(
            {
                "error": "Error consultando productos.",
                "detail": str(exc),
            }
        ), 500


@app.route("/products/<code>", methods=["GET"])
def get_product_by_code(code: str):
    """
    Busca un producto por código.
    """
    try:
        product = find_product_by_code(code)

        if product is None:
            return jsonify(
                {
                    "error": "Producto no encontrado.",
                    "code": code,
                }
            ), 404

        return jsonify(product), 200

    except Exception as exc:
        return jsonify(
            {
                "error": "Error consultando el producto.",
                "detail": str(exc),
            }
        ), 500


if __name__ == "__main__":
    initialize_database()

    debug_enabled = os.getenv("FLASK_DEBUG", "1").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }

    app.run(
        host=os.getenv("FLASK_HOST", "127.0.0.1"),
        port=int(os.getenv("FLASK_PORT", "5000")),
        debug=debug_enabled,
    )
