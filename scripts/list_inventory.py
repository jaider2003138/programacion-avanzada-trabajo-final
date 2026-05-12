"""
Lista productos registrados en el inventario.

Ejecutar:

python scripts/list_inventory.py
"""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from app.services.inventory_service import list_products


def main() -> None:
    products = list_products()

    if not products:
        print("\nNo hay productos registrados en inventario.")
        return

    print("\nInventario registrado")
    print("=" * 80)

    for product in products:
        print(f"ID: {product['id']}")
        print(f"Código: {product['code']}")
        print(f"Nombre: {product['name']}")
        print(f"Categoría: {product['category']}")
        print(f"Cantidad: {product['quantity']}")
        print(f"Confianza: {product['confidence'] * 100:.2f}%")
        print(f"Fecha: {product['created_at']}")
        print(f"Estado: {product['status']}")
        print("-" * 80)


if __name__ == "__main__":
    main()