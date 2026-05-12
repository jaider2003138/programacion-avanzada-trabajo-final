"""
Servicio de inventario usando SQLite.

Este módulo permite:
- Crear la base de datos.
- Registrar productos.
- Consultar productos.
- Buscar productos por código.
- Obtener el siguiente consecutivo para generar códigos.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from datetime import datetime
from typing import Any


DATABASE_PATH = Path("app/database/inventory.db")


def get_connection() -> sqlite3.Connection:
    """
    Crea y devuelve una conexión a SQLite.
    """
    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def initialize_database() -> None:
    """
    Crea la tabla products si no existe.
    """
    with get_connection() as connection:
        cursor = connection.cursor()

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS products (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                code TEXT UNIQUE NOT NULL,
                name TEXT NOT NULL,
                category TEXT NOT NULL,
                quantity INTEGER NOT NULL DEFAULT 1,
                image_path TEXT NOT NULL,
                confidence REAL NOT NULL,
                created_at TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'activo'
            )
            """
        )

        connection.commit()


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
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
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

        connection.commit()

        product_id = cursor.lastrowid

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
            WHERE code = ?
            """,
            (code,),
        )

        row = cursor.fetchone()

    return dict(row) if row else None