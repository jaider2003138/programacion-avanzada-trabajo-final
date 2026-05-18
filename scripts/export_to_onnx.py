"""
Exporta trained_models/product_classifier.keras a ONNX.

Ejecutar desde la raíz del proyecto:
    python scripts/export_to_onnx.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import onnx
import tensorflow as tf
import tf2onnx

MODEL_PATH = Path("trained_models/product_classifier.keras")
ONNX_PATH = Path("trained_models/product_classifier.onnx")


def main() -> int:
    if not MODEL_PATH.exists():
        print(f"Error: no se encontró el modelo en {MODEL_PATH}", file=sys.stderr)
        return 1

    print(f"Cargando {MODEL_PATH} ...")
    model = tf.keras.models.load_model(MODEL_PATH)

    print("Exportando a ONNX (opset 13) ...")
    input_spec = (tf.TensorSpec((None, 224, 224, 3), tf.float32, name="input"),)
    model_proto, _ = tf2onnx.convert.from_keras(
        model,
        input_signature=input_spec,
        opset=13,
    )

    ONNX_PATH.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model_proto, str(ONNX_PATH))
    size_mb = ONNX_PATH.stat().st_size / (1024 * 1024)
    print(f"Modelo ONNX guardado en: {ONNX_PATH} ({size_mb:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
