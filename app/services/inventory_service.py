"""
Servicio de inventario usando PostgreSQL (Supabase).

Este módulo permite:
- Crear la base de datos.
- Registrar productos.
- Consultar productos.
- Buscar productos por código.
- Obtener el siguiente consecutivo para generar códigos.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from datetime import datetime
from typing import Any

import psycopg2
import psycopg2.extras
from dotenv import load_dotenv

load_dotenv()

_DATABASE_URL: str | None = None


def _get_database_url() -> str:
    global _DATABASE_URL
    if _DATABASE_URL is None:
        url = os.getenv("DATABASE_URL")
        if not url:
            raise RuntimeError(
                "DATABASE_URL no está configurada. "
                "Crea un archivo .env con DATABASE_URL=postgresql://..."
            )
        _DATABASE_URL = url
    return _DATABASE_URL


@contextmanager
def get_connection():
    """
    Context manager que abre y cierra una conexión a PostgreSQL.
    """
    connection = psycopg2.connect(
        _get_database_url(),
        cursor_factory=psycopg2.extras.RealDictCursor,
    )
    try:
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def initialize_database() -> None:
    """
    Crea la tabla products si no existe.
    """
    with get_connection() as connection:
        cursor = connection.cursor()

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS products (
                id               SERIAL PRIMARY KEY,
                code             TEXT UNIQUE NOT NULL,
                name             TEXT NOT NULL,
                category         TEXT NOT NULL,
                quantity         INTEGER NOT NULL DEFAULT 1,
                image_path       TEXT NOT NULL,
                confidence       DOUBLE PRECISION NOT NULL,
                created_at       TEXT NOT NULL,
                status           TEXT NOT NULL DEFAULT 'activo'
            )
            """
        )


def get_next_sequence() -> int:
    """
    Obtiene el siguiente consecutivo global del inventario.
    """
    with get_connection() as connection:
        cursor = connection.cursor()

        cursor.execute("SELECT COUNT(*) AS total FROM products")
        row = cursor.fetchone()

        total = int(row["total"]) if row else 0

    return total + 1


def register_product(
    code: str,
    name: str,
    category: str,
    quantity: int,
    image_path: str,
    confidence: float,
) -> dict[str, Any]:
    """
    Registra un producto en la base de datos.
    """
    initialize_database()

    created_at = datetime.now().isoformat(timespec="seconds")

    with get_connection() as connection:
        cursor = connection.cursor()

        cursor.execute(
            """
            INSERT INTO products (
                code,
                name,
                category,
                quantity,
                image_path,
                confidence,
                created_at,
                status
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id
            """,
            (
                code,
                name,
                category,
                quantity,
                image_path,
                confidence,
                created_at,
                "activo",
            ),
        )

        product_id = cursor.fetchone()["id"]

    return {
        "id": product_id,
        "code": code,
        "name": name,
        "category": category,
        "quantity": quantity,
        "image_path": image_path,
        "confidence": confidence,
        "created_at": created_at,
        "status": "activo",
    }


def list_products() -> list[dict[str, Any]]:
    """
    Lista todos los productos registrados.
    """
    initialize_database()

    with get_connection() as connection:
        cursor = connection.cursor()

        cursor.execute(
            """
            SELECT
                id,
                code,
                name,
                category,
                quantity,
                image_path,
                confidence,
                created_at,
                status
            FROM products
            ORDER BY id DESC
            """
        )

        rows = cursor.fetchall()

    return [dict(row) for row in rows]


def find_product_by_code(code: str) -> dict[str, Any] | None:
    """
    Busca un producto por código.
    """
    initialize_database()

    with get_connection() as connection:
        cursor = connection.cursor()

        cursor.execute(
            """
            SELECT
                id,
                code,
                name,
                category,
                quantity,
                image_path,
                confidence,
                created_at,
                status
            FROM products
            WHERE code = %s
            """,
            (code,),
        )

        row = cursor.fetchone()

    return dict(row) if row else None
