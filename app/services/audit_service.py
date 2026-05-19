"""
Servicio de auditoria para registrar acciones sobre el inventario.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from typing import Any

from app.services.inventory_service import get_connection
from app.services.user_service import CREATE_USERS_TABLE_SQL


DEFAULT_USER_NAME = "Usuario local"
DEFAULT_USER_EMAIL = "local@app.com"
DEFAULT_USER_ROLE = "bodega"

CREATE_AUDIT_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS inventory_audit_logs (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES app_users(id),
    user_name TEXT NOT NULL,
    user_email TEXT,
    user_role TEXT NOT NULL,
    action TEXT NOT NULL,
    product_code TEXT,
    product_name TEXT,
    category TEXT,
    quantity INTEGER,
    confidence DOUBLE PRECISION,
    details TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
"""


def _ensure_audit_schema(cursor: Any) -> None:
    """
    Crea o migra las tablas necesarias para auditoria.
    """
    cursor.execute(CREATE_USERS_TABLE_SQL)
    cursor.execute(CREATE_AUDIT_TABLE_SQL)
    cursor.execute(
        """
        ALTER TABLE inventory_audit_logs
        ADD COLUMN IF NOT EXISTS user_id INTEGER REFERENCES app_users(id)
        """
    )
    cursor.execute(
        """
        ALTER TABLE inventory_audit_logs
        ADD COLUMN IF NOT EXISTS user_role TEXT
        """
    )
    cursor.execute(
        """
        UPDATE inventory_audit_logs
        SET user_role = %s
        WHERE user_role IS NULL OR trim(user_role) = ''
        """,
        (DEFAULT_USER_ROLE,),
    )
    cursor.execute(
        """
        ALTER TABLE inventory_audit_logs
        ALTER COLUMN user_role SET DEFAULT 'bodega'
        """
    )
    cursor.execute(
        """
        ALTER TABLE inventory_audit_logs
        ALTER COLUMN user_role SET NOT NULL
        """
    )


def initialize_audit_table() -> None:
    """
    Crea o migra la tabla de auditoria si no existe.
    """
    with get_connection() as connection:
        cursor = connection.cursor()
        _ensure_audit_schema(cursor)


def _clean_text(value: Any) -> str | None:
    if value is None:
        return None

    cleaned = str(value).strip()
    return cleaned or None


def _normalize_details(details: Any) -> str | None:
    if details is None:
        return None

    if isinstance(details, str):
        return details.strip() or None

    return json.dumps(details, ensure_ascii=False, default=str)


def _parse_datetime_filter(value: str | None, field_name: str) -> datetime | None:
    if not value:
        return None

    try:
        if len(value) == 10:
            return datetime.strptime(value, "%Y-%m-%d")
        return datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(
            f"El filtro {field_name} debe tener formato YYYY-MM-DD o ISO datetime."
        ) from exc


def log_inventory_action(
    user_name: str | None,
    user_email: str | None,
    action: str,
    user_id: int | None = None,
    user_role: str | None = None,
    product_code: str | None = None,
    product_name: str | None = None,
    category: str | None = None,
    quantity: int | None = None,
    confidence: float | None = None,
    details: Any = None,
    *,
    raise_errors: bool = False,
) -> bool:
    """
    Guarda una accion de auditoria.

    Por defecto no propaga errores para evitar que un fallo de auditoria bloquee
    la operacion principal del inventario.
    """
    try:
        clean_user_name = _clean_text(user_name) or DEFAULT_USER_NAME
        clean_user_email = _clean_text(user_email) or DEFAULT_USER_EMAIL
        clean_user_role = _clean_text(user_role) or DEFAULT_USER_ROLE
        clean_user_id = int(user_id) if user_id not in (None, "") else None

        with get_connection() as connection:
            cursor = connection.cursor()
            _ensure_audit_schema(cursor)
            cursor.execute(
                """
                INSERT INTO inventory_audit_logs (
                    user_id,
                    user_name,
                    user_email,
                    user_role,
                    action,
                    product_code,
                    product_name,
                    category,
                    quantity,
                    confidence,
                    details
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    clean_user_id,
                    clean_user_name,
                    clean_user_email,
                    clean_user_role,
                    _clean_text(action) or "accion_no_especificada",
                    _clean_text(product_code),
                    _clean_text(product_name),
                    _clean_text(category),
                    quantity,
                    confidence,
                    _normalize_details(details),
                ),
            )
        return True
    except Exception:
        if raise_errors:
            raise
        return False


def list_audit_logs(
    *,
    user_id: int | None = None,
    user_name: str | None = None,
    user_email: str | None = None,
    user_role: str | None = None,
    action: str | None = None,
    product_code: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    limit: int = 100,
) -> list[dict[str, Any]]:
    """
    Consulta registros de auditoria ordenados del mas reciente al mas antiguo.
    """
    start_date = _parse_datetime_filter(date_from, "date_from")
    end_date = _parse_datetime_filter(date_to, "date_to")

    safe_limit = max(1, min(int(limit), 500))
    filters: list[str] = []
    params: list[Any] = []

    if user_name:
        filters.append("user_name ILIKE %s")
        params.append(f"%{user_name.strip()}%")

    if user_id:
        filters.append("user_id = %s")
        params.append(int(user_id))

    if user_email:
        filters.append("user_email ILIKE %s")
        params.append(f"%{user_email.strip()}%")

    if user_role:
        filters.append("user_role = %s")
        params.append(user_role.strip())

    if action:
        filters.append("action = %s")
        params.append(action.strip())

    if product_code:
        filters.append("product_code ILIKE %s")
        params.append(f"%{product_code.strip()}%")

    if start_date:
        filters.append("created_at >= %s")
        params.append(start_date)

    if end_date:
        if date_to and len(date_to) == 10:
            filters.append("created_at < %s")
            params.append(end_date + timedelta(days=1))
        else:
            filters.append("created_at <= %s")
            params.append(end_date)

    where_clause = f"WHERE {' AND '.join(filters)}" if filters else ""

    query = f"""
        SELECT
            id,
            user_id,
            user_name,
            user_email,
            user_role,
            action,
            product_code,
            product_name,
            category,
            quantity,
            confidence,
            details,
            created_at
        FROM inventory_audit_logs
        {where_clause}
        ORDER BY created_at DESC, id DESC
        LIMIT %s
    """

    with get_connection() as connection:
        cursor = connection.cursor()
        _ensure_audit_schema(cursor)
        cursor.execute(query, [*params, safe_limit])
        rows = cursor.fetchall()

    logs: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        created_at = item.get("created_at")
        if hasattr(created_at, "isoformat"):
            item["created_at"] = created_at.isoformat()
        logs.append(item)

    return logs
