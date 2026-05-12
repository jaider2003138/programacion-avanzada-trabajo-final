"""
API Flask para el Sistema Inteligente de Clasificación e Inventariado
de Productos de Despensa.

Ejecutar desde la raíz del proyecto:

    python app/api/app.py
"""

from __future__ import annotations

import sys
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

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True,
    )