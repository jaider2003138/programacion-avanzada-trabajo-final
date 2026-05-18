"""
Migra los registros de app/database/inventory.db (SQLite) a PostgreSQL.

Ejecutar desde la raíz del proyecto:
    python scripts/migrate_to_postgres.py
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import psycopg2
import psycopg2.extras
from dotenv import load_dotenv
import os

load_dotenv()

SQLITE_PATH = Path("app/database/inventory.db")


def read_sqlite_records() -> list[dict]:
    if not SQLITE_PATH.exists():
        print(f"No se encontró la BD SQLite en {SQLITE_PATH} — nada que migrar.")
        return []

    con = sqlite3.connect(SQLITE_PATH)
    con.row_factory = sqlite3.Row
    rows = con.execute(
        "SELECT code, name, category, quantity, image_path, confidence, created_at, status "
        "FROM products"
    ).fetchall()
    con.close()
    return [dict(r) for r in rows]


def migrate(records: list[dict]) -> int:
    url = os.getenv("DATABASE_URL")
    if not url:
        raise RuntimeError("DATABASE_URL no está configurada en .env")

    connection = psycopg2.connect(url, cursor_factory=psycopg2.extras.RealDictCursor)
    try:
        cursor = connection.cursor()
        inserted = 0

        for record in records:
            cursor.execute(
                """
                INSERT INTO products (
                    code, name, category, quantity,
                    image_path, confidence, created_at, status
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (code) DO NOTHING
                """,
                (
                    record["code"],
                    record["name"],
                    record["category"],
                    record["quantity"],
                    record["image_path"],
                    record["confidence"],
                    record["created_at"],
                    record["status"],
                ),
            )
            if cursor.rowcount > 0:
                inserted += 1

        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()

    return inserted


def main() -> int:
    records = read_sqlite_records()
    total = len(records)

    if total == 0:
        print("Sin registros que migrar.")
        return 0

    print(f"Registros encontrados en SQLite: {total}")
    inserted = migrate(records)
    skipped = total - inserted
    print(f"Migrados a PostgreSQL : {inserted}")
    print(f"Omitidos (ya existían): {skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
