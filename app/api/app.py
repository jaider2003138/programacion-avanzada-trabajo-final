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
from app.services.audit_service import (
    DEFAULT_USER_EMAIL,
    DEFAULT_USER_NAME,
    DEFAULT_USER_ROLE,
    initialize_audit_table,
    list_audit_logs,
    log_inventory_action,
)
from app.services.code_generator import generate_inventory_code
from app.services.inventory_service import (
    find_product_by_code,
    get_next_sequence,
    initialize_database,
    list_products,
    register_product,
    update_product,
)
from app.services.user_service import (
    authenticate_user,
    count_users,
    create_user,
    get_user_by_email,
    get_user_by_id,
    initialize_users_table,
    list_users as list_app_users,
    update_user_status,
)


app = Flask(__name__)
CORS(app)


def _clean_request_text(value: Any, default: str | None = None) -> str | None:
    if value is None:
        return default

    cleaned = str(value).strip()
    return cleaned or default


def _get_request_user(data: dict[str, Any] | None = None) -> dict[str, Any]:
    source = data or {}
    user_id = (
        request.headers.get("X-User-Id")
        or request.form.get("user_id")
        or request.args.get("user_id")
        or source.get("user_id")
    )
    user_name = (
        request.headers.get("X-User-Name")
        or request.form.get("user_name")
        or request.args.get("user_name")
        or source.get("user_name")
    )
    user_email = (
        request.headers.get("X-User-Email")
        or request.form.get("user_email")
        or request.args.get("user_email")
        or source.get("user_email")
    )
    user_role = (
        request.headers.get("X-User-Role")
        or request.form.get("user_role")
        or request.args.get("user_role")
        or source.get("user_role")
    )

    database_user = None
    if user_id not in (None, ""):
        try:
            database_user = get_user_by_id(int(user_id))
        except (TypeError, ValueError):
            database_user = None
    elif user_email:
        database_user = get_user_by_email(str(user_email))

    if database_user and database_user.get("status") == "activo":
        return {
            "user_id": database_user["id"],
            "user_name": database_user["full_name"],
            "user_email": database_user["email"],
            "user_role": database_user["role"],
        }

    return {
        "user_id": None,
        "user_name": _clean_request_text(user_name, DEFAULT_USER_NAME) or DEFAULT_USER_NAME,
        "user_email": _clean_request_text(user_email, DEFAULT_USER_EMAIL) or DEFAULT_USER_EMAIL,
        "user_role": _clean_request_text(user_role, DEFAULT_USER_ROLE) or DEFAULT_USER_ROLE,
    }


def _require_roles(user: dict[str, Any], allowed_roles: set[str]):
    if not user.get("user_id"):
        return jsonify({"error": "Debes iniciar sesion para realizar esta accion."}), 401

    if user.get("user_role") not in allowed_roles:
        return jsonify(
            {
                "error": "No tienes permisos para realizar esta accion.",
                "required_roles": sorted(allowed_roles),
                "current_role": user.get("user_role"),
            }
        ), 403

    return None


def _audit_user_kwargs(user: dict[str, Any]) -> dict[str, Any]:
    return {
        "user_id": user.get("user_id"),
        "user_name": user.get("user_name"),
        "user_email": user.get("user_email"),
        "user_role": user.get("user_role"),
    }


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
                "POST /auth/login",
                "POST /users/bootstrap",
                "GET /users",
                "POST /users",
                "POST /predict",
                "POST /predict/batch",
                "POST /products",
                "GET /products",
                "GET /products/<code>",
                "PATCH /products/<code>",
                "DELETE /products/<code>",
                "GET /audit/logs",
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


@app.route("/auth/login", methods=["POST"])
def login():
    """
    Inicia sesion con email y contrasena.
    """
    data: dict[str, Any] = request.get_json(silent=True) or {}
    email = str(data.get("email", "")).strip()
    password = str(data.get("password", ""))

    if not email or not password:
        return jsonify({"error": "Email y contrasena son obligatorios."}), 400

    user = authenticate_user(email, password)
    if user is None:
        return jsonify({"error": "Credenciales invalidas o usuario inactivo."}), 401

    return jsonify({"message": "Inicio de sesion correcto.", "user": user}), 200


@app.route("/users/bootstrap", methods=["POST"])
def bootstrap_admin():
    """
    Crea el primer administrador si no existen usuarios.
    """
    if count_users() > 0:
        return jsonify({"error": "Ya existen usuarios registrados."}), 403

    data: dict[str, Any] = request.get_json(silent=True) or {}
    required_fields = ["full_name", "email", "password"]
    missing_fields = [field for field in required_fields if not data.get(field)]

    if missing_fields:
        return jsonify(
            {
                "error": "Faltan campos obligatorios.",
                "missing_fields": missing_fields,
            }
        ), 400

    try:
        user = create_user(
            full_name=str(data["full_name"]),
            email=str(data["email"]),
            password=str(data["password"]),
            role="admin",
            status="activo",
        )
        return jsonify({"message": "Administrador inicial creado.", "user": user}), 201
    except Exception as exc:
        return jsonify({"error": "No se pudo crear el administrador.", "detail": str(exc)}), 500


@app.route("/users", methods=["GET"])
def get_users():
    """
    Lista usuarios. Solo admin.
    """
    user = _get_request_user()
    auth_error = _require_roles(user, {"admin"})
    if auth_error:
        return auth_error

    try:
        users = list_app_users()
        return jsonify({"total": len(users), "users": users}), 200
    except Exception as exc:
        return jsonify({"error": "Error consultando usuarios.", "detail": str(exc)}), 500


@app.route("/users", methods=["POST"])
def post_user():
    """
    Crea usuarios. Solo admin.
    """
    data: dict[str, Any] = request.get_json(silent=True) or {}
    user = _get_request_user(data)
    auth_error = _require_roles(user, {"admin"})
    if auth_error:
        return auth_error

    required_fields = ["full_name", "email", "password", "role"]
    missing_fields = [field for field in required_fields if not data.get(field)]
    if missing_fields:
        return jsonify(
            {
                "error": "Faltan campos obligatorios.",
                "missing_fields": missing_fields,
            }
        ), 400

    user = _get_request_user(data)
    auth_error = _require_roles(user, {"admin", "bodega"})
    if auth_error:
        return auth_error

    try:
        created_user = create_user(
            full_name=str(data["full_name"]),
            email=str(data["email"]),
            password=str(data["password"]),
            role=str(data["role"]),
            status=str(data.get("status", "activo")),
        )
        log_inventory_action(
            **_audit_user_kwargs(user),
            action="usuario_creado",
            details={
                "created_user_id": created_user["id"],
                "created_user_email": created_user["email"],
                "created_user_role": created_user["role"],
            },
        )
        return jsonify({"message": "Usuario creado correctamente.", "user": created_user}), 201
    except ValueError as exc:
        return jsonify({"error": "Datos invalidos.", "detail": str(exc)}), 400
    except Exception as exc:
        return jsonify({"error": "No se pudo crear el usuario.", "detail": str(exc)}), 500


@app.route("/users/<int:user_id>/status", methods=["PATCH"])
def patch_user_status(user_id: int):
    """
    Cambia el estado de un usuario. Solo admin.
    """
    data: dict[str, Any] = request.get_json(silent=True) or {}
    user = _get_request_user(data)
    auth_error = _require_roles(user, {"admin"})
    if auth_error:
        return auth_error

    status = str(data.get("status", "")).strip().lower()
    if not status:
        return jsonify({"error": "El campo status es obligatorio."}), 400

    try:
        updated_user = update_user_status(user_id, status)
        if updated_user is None:
            return jsonify({"error": "Usuario no encontrado."}), 404

        log_inventory_action(
            **_audit_user_kwargs(user),
            action="usuario_actualizado",
            details={
                "updated_user_id": updated_user["id"],
                "updated_user_email": updated_user["email"],
                "status": updated_user["status"],
            },
        )
        return jsonify({"message": "Usuario actualizado.", "user": updated_user}), 200
    except Exception as exc:
        return jsonify({"error": "No se pudo actualizar el usuario.", "detail": str(exc)}), 500


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
    user = _get_request_user()
    auth_error = _require_roles(user, {"admin", "bodega"})
    if auth_error:
        return auth_error

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

        log_inventory_action(
            **_audit_user_kwargs(user),
            action="producto_clasificado",
            product_code=generated_code,
            product_name=Path(image_file.filename or "").stem or None,
            category=prediction["predicted_category"],
            confidence=float(prediction["confidence"]),
            details={
                "filename": image_file.filename,
                "confidence_percent": prediction.get("confidence_percent"),
                "confidence_level": prediction.get("confidence_level"),
                "is_classifiable": prediction.get("is_classifiable", True),
            },
        )

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
    user = _get_request_user()
    auth_error = _require_roles(user, {"admin", "bodega"})
    if auth_error:
        return auth_error

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
                    "confidence_level": prediction.get("confidence_level"),
                    "top_predictions": prediction["top_predictions"],
                    "generated_code": generated_code,
                    "is_classifiable": prediction.get("is_classifiable", True),
                    "classification_warning": prediction.get("classification_warning"),
                }
            )
        except Exception as exc:
            total_skipped += 1
            results.append({"filename": filename, "error": str(exc)})

    successful_results = [result for result in results if "error" not in result]
    avg_confidence = None
    if successful_results:
        avg_confidence = sum(
            float(result.get("confidence", 0))
            for result in successful_results
        ) / len(successful_results)

    for result in successful_results:
        log_inventory_action(
            **_audit_user_kwargs(user),
            action="producto_clasificado",
            product_code=result.get("generated_code"),
            product_name=Path(str(result.get("filename", ""))).stem or None,
            category=result.get("predicted_category"),
            confidence=float(result.get("confidence", 0)),
            details={
                "filename": result.get("filename"),
                "source": "cargue_masivo",
                "confidence_percent": result.get("confidence_percent"),
                "confidence_level": result.get("confidence_level"),
                "is_classifiable": result.get("is_classifiable", True),
            },
        )

    log_inventory_action(
        **_audit_user_kwargs(user),
        action="cargue_masivo_realizado",
        quantity=total_received - total_skipped,
        confidence=avg_confidence,
        details={
            "source": "zip" if has_zip else "files",
            "total_received": total_received,
            "total_processed": total_received - total_skipped,
            "total_skipped": total_skipped,
        },
    )

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

    user = _get_request_user(data)
    auth_error = _require_roles(user, {"admin", "bodega"})
    if auth_error:
        return auth_error

    try:
        product = register_product(
            code=str(data["code"]),
            name=str(data["name"]),
            category=str(data["category"]),
            quantity=int(data["quantity"]),
            image_path=str(data["image_path"]),
            confidence=float(data["confidence"]),
        )

        log_inventory_action(
            **_audit_user_kwargs(user),
            action="producto_agregado",
            product_code=product["code"],
            product_name=product["name"],
            category=product["category"],
            quantity=int(product["quantity"]),
            confidence=float(product["confidence"]),
            details=data.get("audit_details") or "Producto registrado en inventario.",
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
    user = _get_request_user()
    auth_error = _require_roles(user, {"admin", "bodega"})
    if auth_error:
        return auth_error

    try:
        products = list_products()

        if request.args.get("audit", "").lower() in {"1", "true", "yes"}:
            log_inventory_action(
                **_audit_user_kwargs(user),
                action="inventario_consultado",
                quantity=len(products),
                details="Consulta del listado de inventario.",
            )

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


@app.route("/products/<code>", methods=["PATCH"])
def patch_product(code: str):
    """
    Actualiza campos permitidos de un producto.
    """
    data: dict[str, Any] = request.get_json(silent=True) or {}
    user = _get_request_user(data)
    auth_error = _require_roles(user, {"admin"})
    if auth_error:
        return auth_error

    allowed_fields = {
        "name",
        "category",
        "quantity",
        "image_path",
        "confidence",
        "status",
    }
    updates = {
        field: data[field]
        for field in allowed_fields
        if field in data
    }

    if not updates:
        return jsonify(
            {
                "error": "No se enviaron campos actualizables.",
                "allowed_fields": sorted(allowed_fields),
            }
        ), 400

    try:
        product = update_product(code, **updates)

        if product is None:
            return jsonify(
                {
                    "error": "Producto no encontrado.",
                    "code": code,
                }
            ), 404

        log_inventory_action(
            **_audit_user_kwargs(user),
            action="producto_actualizado",
            product_code=product["code"],
            product_name=product["name"],
            category=product["category"],
            quantity=int(product["quantity"]),
            confidence=float(product["confidence"]),
            details={
                "updated_fields": sorted(updates.keys()),
                "source": data.get("audit_details") or "Actualizacion de producto.",
            },
        )

        return jsonify(
            {
                "message": "Producto actualizado correctamente.",
                "product": product,
            }
        ), 200

    except ValueError as exc:
        return jsonify(
            {
                "error": "Datos invalidos para actualizar el producto.",
                "detail": str(exc),
            }
        ), 400
    except Exception as exc:
        return jsonify(
            {
                "error": "Error actualizando el producto.",
                "detail": str(exc),
            }
        ), 500


@app.route("/products/<code>", methods=["DELETE"])
def delete_product(code: str):
    """
    Desactiva un producto. Solo admin.
    """
    data: dict[str, Any] = request.get_json(silent=True) or {}
    user = _get_request_user(data)
    auth_error = _require_roles(user, {"admin"})
    if auth_error:
        return auth_error

    try:
        product = update_product(
            code,
            status="inactivo",
        )

        if product is None:
            return jsonify(
                {
                    "error": "Producto no encontrado.",
                    "code": code,
                }
            ), 404

        log_inventory_action(
            **_audit_user_kwargs(user),
            action="producto_desactivado",
            product_code=product["code"],
            product_name=product["name"],
            category=product["category"],
            quantity=int(product["quantity"]),
            confidence=float(product["confidence"]),
            details="Producto desactivado desde la API.",
        )

        return jsonify(
            {
                "message": "Producto desactivado correctamente.",
                "product": product,
            }
        ), 200

    except Exception as exc:
        return jsonify(
            {
                "error": "Error desactivando el producto.",
                "detail": str(exc),
            }
        ), 500


@app.route("/products/<code>", methods=["GET"])
def get_product_by_code(code: str):
    """
    Busca un producto por código.
    """
    user = _get_request_user()
    auth_error = _require_roles(user, {"admin", "bodega"})
    if auth_error:
        return auth_error

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


@app.route("/audit/logs", methods=["GET"])
def get_audit_logs():
    """
    Lista registros de auditoria con filtros opcionales.
    """
    user = _get_request_user()
    auth_error = _require_roles(user, {"admin", "bodega"})
    if auth_error:
        return auth_error

    try:
        limit = int(request.args.get("limit", "100"))
        requested_user_id = request.args.get("user_id")
        if user["user_role"] == "bodega":
            requested_user_id = user["user_id"]

        logs = list_audit_logs(
            user_id=requested_user_id,
            user_name=request.args.get("user_name"),
            user_email=request.args.get("user_email"),
            user_role=request.args.get("user_role"),
            action=request.args.get("action"),
            product_code=request.args.get("product_code"),
            date_from=request.args.get("date_from"),
            date_to=request.args.get("date_to"),
            limit=limit,
        )

        return jsonify(
            {
                "total": len(logs),
                "logs": logs,
            }
        ), 200

    except ValueError as exc:
        return jsonify(
            {
                "error": "Filtros invalidos para consultar auditoria.",
                "detail": str(exc),
            }
        ), 400
    except Exception as exc:
        return jsonify(
            {
                "error": "Error consultando logs de auditoria.",
                "detail": str(exc),
            }
        ), 500


if __name__ == "__main__":
    initialize_database()
    initialize_users_table()
    initialize_audit_table()

    debug_enabled = os.getenv("FLASK_DEBUG", "1").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }

    app.run(
    host=os.getenv("FLASK_HOST", "0.0.0.0"),
    port=int(os.getenv("PORT", os.getenv("FLASK_PORT", "5000"))),
    debug=debug_enabled,
    )
