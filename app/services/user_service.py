"""
Servicio de usuarios y roles de la aplicacion.
"""

from __future__ import annotations

from typing import Any

from werkzeug.security import check_password_hash, generate_password_hash

from app.services.inventory_service import get_connection


VALID_ROLES = {"admin", "bodega"}

CREATE_USERS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS app_users (
    id SERIAL PRIMARY KEY,
    full_name TEXT NOT NULL,
    email TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    role TEXT NOT NULL CHECK (role IN ('admin', 'bodega')),
    status TEXT NOT NULL DEFAULT 'activo',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
"""


def initialize_users_table() -> None:
    """
    Crea la tabla de usuarios si no existe.
    """
    with get_connection() as connection:
        cursor = connection.cursor()
        cursor.execute(CREATE_USERS_TABLE_SQL)


def _serialize_user(row: Any) -> dict[str, Any]:
    user = dict(row)
    user.pop("password_hash", None)

    created_at = user.get("created_at")
    if hasattr(created_at, "isoformat"):
        user["created_at"] = created_at.isoformat()

    return user


def count_users() -> int:
    """
    Cuenta usuarios registrados.
    """
    initialize_users_table()

    with get_connection() as connection:
        cursor = connection.cursor()
        cursor.execute("SELECT COUNT(*) AS total FROM app_users")
        row = cursor.fetchone()

    return int(row["total"]) if row else 0


def get_user_by_id(user_id: int) -> dict[str, Any] | None:
    """
    Busca un usuario por id.
    """
    initialize_users_table()

    with get_connection() as connection:
        cursor = connection.cursor()
        cursor.execute(
            """
            SELECT
                id,
                full_name,
                email,
                password_hash,
                role,
                status,
                created_at
            FROM app_users
            WHERE id = %s
            """,
            (int(user_id),),
        )
        row = cursor.fetchone()

    return dict(row) if row else None


def get_user_by_email(email: str) -> dict[str, Any] | None:
    """
    Busca un usuario por correo.
    """
    initialize_users_table()

    with get_connection() as connection:
        cursor = connection.cursor()
        cursor.execute(
            """
            SELECT
                id,
                full_name,
                email,
                password_hash,
                role,
                status,
                created_at
            FROM app_users
            WHERE lower(email) = lower(%s)
            """,
            (email.strip(),),
        )
        row = cursor.fetchone()

    return dict(row) if row else None


def authenticate_user(email: str, password: str) -> dict[str, Any] | None:
    """
    Valida email y contrasena contra la tabla app_users.
    """
    user = get_user_by_email(email)
    if not user:
        return None

    if user.get("status") != "activo":
        return None

    if not check_password_hash(str(user["password_hash"]), password):
        return None

    return _serialize_user(user)


def create_user(
    full_name: str,
    email: str,
    password: str,
    role: str,
    status: str = "activo",
) -> dict[str, Any]:
    """
    Crea un usuario con contrasena hasheada.
    """
    initialize_users_table()

    clean_role = role.strip().lower()
    if clean_role not in VALID_ROLES:
        raise ValueError("El rol debe ser 'admin' o 'bodega'.")

    clean_status = status.strip().lower() or "activo"
    password_hash = generate_password_hash(password)

    with get_connection() as connection:
        cursor = connection.cursor()
        cursor.execute(
            """
            INSERT INTO app_users (
                full_name,
                email,
                password_hash,
                role,
                status
            )
            VALUES (%s, %s, %s, %s, %s)
            RETURNING
                id,
                full_name,
                email,
                role,
                status,
                created_at
            """,
            (
                full_name.strip(),
                email.strip().lower(),
                password_hash,
                clean_role,
                clean_status,
            ),
        )
        row = cursor.fetchone()

    return _serialize_user(row)


def list_users() -> list[dict[str, Any]]:
    """
    Lista usuarios sin exponer hashes de contrasena.
    """
    initialize_users_table()

    with get_connection() as connection:
        cursor = connection.cursor()
        cursor.execute(
            """
            SELECT
                id,
                full_name,
                email,
                role,
                status,
                created_at
            FROM app_users
            ORDER BY id DESC
            """
        )
        rows = cursor.fetchall()

    return [_serialize_user(row) for row in rows]


def update_user_status(user_id: int, status: str) -> dict[str, Any] | None:
    """
    Actualiza el estado de un usuario.
    """
    initialize_users_table()

    with get_connection() as connection:
        cursor = connection.cursor()
        cursor.execute(
            """
            UPDATE app_users
            SET status = %s
            WHERE id = %s
            RETURNING
                id,
                full_name,
                email,
                role,
                status,
                created_at
            """,
            (status.strip().lower(), int(user_id)),
        )
        row = cursor.fetchone()

    return _serialize_user(row) if row else None
