"""
Servicio para generar códigos únicos de inventario.

Formato del código:

INV-CAT-YYYYMM-0001

Ejemplos:
INV-ARG-202605-0001
INV-PAS-202605-0002
INV-CAF-202605-0003
"""

from __future__ import annotations

from datetime import datetime


CATEGORY_PREFIXES = {
    "arroz_y_granos": "ARG",
    "pastas": "PAS",
    "aceites": "ACE",
    "salsas_y_condimentos": "SAL",
    "cafe_chocolate": "CAF",
    "enlatados": "ENL",
    "azucar_sal": "AZS",
}


def get_category_prefix(category: str) -> str:
    """
    Devuelve el prefijo de una categoría.
    """
    if category not in CATEGORY_PREFIXES:
        return "GEN"

    return CATEGORY_PREFIXES[category]


def generate_inventory_code(category: str, sequence_number: int) -> str:
    """
    Genera un código único estructurado para el inventario.

    Parámetros:
    - category: categoría predicha por el modelo.
    - sequence_number: consecutivo del producto.

    Retorna:
    - Código tipo INV-CAF-202605-0001.
    """
    prefix = get_category_prefix(category)
    current_date = datetime.now().strftime("%Y%m")
    sequence = str(sequence_number).zfill(4)

    return f"INV-{prefix}-{current_date}-{sequence}"