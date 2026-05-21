"""
Evalúa el modelo entrenado usando el dataset de test.

Genera:
- trained_models/evaluation_report.json
- trained_models/confusion_matrix.png
- trained_models/test_metrics_by_class.png
- trained_models/prediction_distribution.png

Ejemplo de uso:

python scripts/evaluate_model.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image, ImageOps
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))


DEFAULT_MODEL_PATH = Path("trained_models/product_classifier.keras")
DEFAULT_LABELS_PATH = Path("datasets/processed/labels.json")
DEFAULT_TEST_DIR = Path("datasets/processed/test")
DEFAULT_OUTPUT_DIR = Path("trained_models")

IMAGE_SIZE = (224, 224)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluar el modelo de clasificación con el dataset de test."
    )

    parser.add_argument(
        "--model",
        default=str(DEFAULT_MODEL_PATH),
        help="Ruta del modelo Keras entrenado.",
    )

    parser.add_argument(
        "--labels",
        default=str(DEFAULT_LABELS_PATH),
        help="Ruta del archivo labels.json.",
    )

    parser.add_argument(
        "--test-dir",
        default=str(DEFAULT_TEST_DIR),
        help="Carpeta del dataset de test.",
    )

    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Carpeta donde se guardarán los reportes.",
    )

    return parser.parse_args()


def load_labels(labels_path: Path) -> list[str]:
    if not labels_path.exists():
        raise FileNotFoundError(f"No se encontró labels.json: {labels_path}")

    with labels_path.open("r", encoding="utf-8") as file:
        labels_dict = json.load(file)

    return [labels_dict[str(index)] for index in range(len(labels_dict))]


def load_image(image_path: Path) -> np.ndarray:
    image = Image.open(image_path)
    image = ImageOps.exif_transpose(image)
    image = image.convert("RGB")
    image = image.resize(IMAGE_SIZE)

    image_array = np.array(image, dtype=np.float32)
    image_array = np.expand_dims(image_array, axis=0)

    return image_array


def collect_test_images(test_dir: Path, class_names: list[str]) -> list[tuple[Path, str]]:
    if not test_dir.exists():
        raise FileNotFoundError(f"No se encontró la carpeta de test: {test_dir}")

    samples: list[tuple[Path, str]] = []
    valid_extensions = {".jpg", ".jpeg", ".png", ".webp"}

    for class_name in class_names:
        class_dir = test_dir / class_name

        if not class_dir.exists():
            print(f"Advertencia: no existe carpeta de test para la clase: {class_name}")
            continue

        for image_path in class_dir.rglob("*"):
            if image_path.suffix.lower() in valid_extensions:
                samples.append((image_path, class_name))

    return samples


def predict_dataset(
    model: tf.keras.Model,
    samples: list[tuple[Path, str]],
    class_names: list[str],
) -> tuple[list[str], list[str], list[float], list[dict[str, Any]]]:
    y_true: list[str] = []
    y_pred: list[str] = []
    confidences: list[float] = []
    rows: list[dict[str, Any]] = []

    total = len(samples)

    for index, (image_path, true_label) in enumerate(samples, start=1):
        try:
            image_array = load_image(image_path)
            predictions = model.predict(image_array, verbose=0)[0]

            pred_index = int(np.argmax(predictions))
            predicted_label = class_names[pred_index]
            confidence = float(predictions[pred_index])
            confidence_percent = confidence * 100

            y_true.append(true_label)
            y_pred.append(predicted_label)
            confidences.append(confidence)

            rows.append(
                {
                    "image_path": str(image_path),
                    "true_label": true_label,
                    "predicted_label": predicted_label,
                    "confidence": confidence,
                    "confidence_percent": confidence_percent,
                    "is_correct": true_label == predicted_label,
                }
            )

            if index % 50 == 0 or index == total:
                print(f"Procesadas {index}/{total} imágenes...")

        except Exception as exc:
            rows.append(
                {
                    "image_path": str(image_path),
                    "true_label": true_label,
                    "predicted_label": None,
                    "confidence": None,
                    "confidence_percent": None,
                    "is_correct": False,
                    "error": str(exc),
                }
            )

    return y_true, y_pred, confidences, rows


def calculate_false_counts(
    class_names: list[str],
    y_true: list[str],
    y_pred: list[str],
) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}

    for class_name in class_names:
        false_positives = sum(
            1
            for true_label, predicted_label in zip(y_true, y_pred)
            if predicted_label == class_name and true_label != class_name
        )

        false_negatives = sum(
            1
            for true_label, predicted_label in zip(y_true, y_pred)
            if true_label == class_name and predicted_label != class_name
        )

        true_positives = sum(
            1
            for true_label, predicted_label in zip(y_true, y_pred)
            if true_label == class_name and predicted_label == class_name
        )

        result[class_name] = {
            "true_positives": true_positives,
            "false_positives": false_positives,
            "false_negatives": false_negatives,
        }

    return result


def save_confusion_matrix_plot(
    matrix: np.ndarray,
    class_names: list[str],
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 8))

    im = ax.imshow(matrix)
    ax.figure.colorbar(im, ax=ax)

    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))

    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)

    ax.set_xlabel("Predicción")
    ax.set_ylabel("Clase real")
    ax.set_title("Matriz de confusión")

    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(
                j,
                i,
                str(matrix[i, j]),
                ha="center",
                va="center",
            )

    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def save_metrics_by_class_plot(
    report: dict[str, Any],
    class_names: list[str],
    output_path: Path,
) -> None:
    rows = []

    for class_name in class_names:
        metrics = report.get(class_name, {})
        rows.append(
            {
                "class_name": class_name,
                "precision": metrics.get("precision", 0),
                "recall": metrics.get("recall", 0),
                "f1_score": metrics.get("f1-score", 0),
            }
        )

    df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(11, 6))

    x = np.arange(len(df))
    width = 0.25

    ax.bar(x - width, df["precision"], width, label="Precision")
    ax.bar(x, df["recall"], width, label="Recall")
    ax.bar(x + width, df["f1_score"], width, label="F1-score")

    ax.set_title("Métricas por categoría")
    ax.set_xlabel("Categoría")
    ax.set_ylabel("Valor")
    ax.set_ylim(0, 1)
    ax.set_xticks(x)
    ax.set_xticklabels(df["class_name"], rotation=45, ha="right")
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def save_prediction_distribution_plot(
    class_names: list[str],
    y_pred: list[str],
    output_path: Path,
) -> None:
    counts = {class_name: y_pred.count(class_name) for class_name in class_names}

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(counts.keys(), counts.values())

    ax.set_title("Distribución de predicciones")
    ax.set_xlabel("Categoría predicha")
    ax.set_ylabel("Cantidad de imágenes")
    ax.tick_params(axis="x", rotation=45)

    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def save_results_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False, encoding="utf-8")


def main() -> None:
    args = parse_args()

    model_path = Path(args.model)
    labels_path = Path(args.labels)
    test_dir = Path(args.test_dir)
    output_dir = Path(args.output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    if not model_path.exists():
        raise FileNotFoundError(f"No se encontró el modelo: {model_path}")

    print("Cargando etiquetas...")
    class_names = load_labels(labels_path)

    print("Recolectando imágenes de test...")
    samples = collect_test_images(test_dir, class_names)

    if not samples:
        raise RuntimeError(
            f"No se encontraron imágenes de test en {test_dir}. "
            "Ejecuta primero el script de preparación del dataset."
        )

    print(f"Total de imágenes de test encontradas: {len(samples)}")

    print("Cargando modelo...")
    model = tf.keras.models.load_model(model_path)

    print("Evaluando modelo...")
    y_true, y_pred, confidences, rows = predict_dataset(model, samples, class_names)

    total_images = len(y_true)
    correct_predictions = sum(
        1 for true_label, predicted_label in zip(y_true, y_pred)
        if true_label == predicted_label
    )
    incorrect_predictions = total_images - correct_predictions
    test_accuracy = accuracy_score(y_true, y_pred)
    average_confidence = (
        sum(confidences) / len(confidences)
        if confidences
        else 0
    )

    report = classification_report(
        y_true,
        y_pred,
        labels=class_names,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=class_names,
    )

    false_counts = calculate_false_counts(class_names, y_true, y_pred)

    evaluation_report = {
        "model_path": str(model_path),
        "test_dir": str(test_dir),
        "total_images": total_images,
        "correct_predictions": correct_predictions,
        "incorrect_predictions": incorrect_predictions,
        "test_accuracy": test_accuracy,
        "test_accuracy_percent": test_accuracy * 100,
        "average_confidence": average_confidence,
        "average_confidence_percent": average_confidence * 100,
        "false_positives_total": sum(
            values["false_positives"] for values in false_counts.values()
        ),
        "false_negatives_total": sum(
            values["false_negatives"] for values in false_counts.values()
        ),
        "per_class_metrics": {},
    }

    for class_name in class_names:
        class_report = report.get(class_name, {})
        class_false_counts = false_counts.get(class_name, {})

        evaluation_report["per_class_metrics"][class_name] = {
            "precision": class_report.get("precision", 0),
            "recall": class_report.get("recall", 0),
            "f1_score": class_report.get("f1-score", 0),
            "support": class_report.get("support", 0),
            "true_positives": class_false_counts.get("true_positives", 0),
            "false_positives": class_false_counts.get("false_positives", 0),
            "false_negatives": class_false_counts.get("false_negatives", 0),
        }

    report_path = output_dir / "evaluation_report.json"
    matrix_path = output_dir / "confusion_matrix.png"
    metrics_plot_path = output_dir / "test_metrics_by_class.png"
    distribution_path = output_dir / "prediction_distribution.png"
    results_csv_path = output_dir / "test_predictions.csv"

    print("Guardando reporte JSON...")
    report_path.write_text(
        json.dumps(evaluation_report, indent=4, ensure_ascii=False),
        encoding="utf-8",
    )

    print("Guardando CSV de predicciones...")
    save_results_csv(rows, results_csv_path)

    print("Guardando matriz de confusión...")
    save_confusion_matrix_plot(matrix, class_names, matrix_path)

    print("Guardando métricas por categoría...")
    save_metrics_by_class_plot(report, class_names, metrics_plot_path)

    print("Guardando distribución de predicciones...")
    save_prediction_distribution_plot(class_names, y_pred, distribution_path)

    print("\nEvaluación finalizada.")
    print("=" * 50)
    print(f"Total imágenes: {total_images}")
    print(f"Correctas: {correct_predictions}")
    print(f"Incorrectas: {incorrect_predictions}")
    print(f"Accuracy: {test_accuracy * 100:.2f}%")
    print(f"Confianza promedio: {average_confidence * 100:.2f}%")
    print(f"Falsos positivos: {evaluation_report['false_positives_total']}")
    print(f"Falsos negativos: {evaluation_report['false_negatives_total']}")
    print("=" * 50)
    print(f"Reporte: {report_path}")
    print(f"Matriz de confusión: {matrix_path}")
    print(f"Métricas por clase: {metrics_plot_path}")
    print(f"Distribución: {distribution_path}")
    print(f"CSV predicciones: {results_csv_path}")


if __name__ == "__main__":
    main()