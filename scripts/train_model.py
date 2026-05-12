"""
Entrenamiento del modelo clasificador de productos de despensa.

Este script:
1. Carga imágenes desde datasets/processed/train, validation y test.
2. Usa MobileNetV2 con Transfer Learning.
3. Entrena una cabeza clasificadora para las clases del proyecto.
4. Evalúa el modelo con el conjunto de prueba.
5. Genera métricas, matriz de confusión y gráficas.
6. Guarda el modelo entrenado en trained_models/product_classifier.keras.

Ejecutar desde la raíz del proyecto:

    python scripts/train_model.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from datetime import datetime

import numpy as np
import tensorflow as tf
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight


# ==========================
# Configuración general
# ==========================

DATASET_DIR = Path("datasets/processed")
TRAIN_DIR = DATASET_DIR / "train"
VALIDATION_DIR = DATASET_DIR / "validation"
TEST_DIR = DATASET_DIR / "test"
LABELS_PATH = DATASET_DIR / "labels.json"

OUTPUT_DIR = Path("trained_models")
MODEL_PATH = OUTPUT_DIR / "product_classifier.keras"
REPORT_PATH = OUTPUT_DIR / "training_report.json"
CONFUSION_MATRIX_PATH = OUTPUT_DIR / "confusion_matrix.png"
ACCURACY_LOSS_PLOT_PATH = OUTPUT_DIR / "accuracy_loss.png"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 15
SEED = 42


# ==========================
# Utilidades
# ==========================

def load_labels(labels_path: Path) -> list[str]:
    """
    Carga labels.json y devuelve las clases ordenadas por índice.

    Ejemplo esperado:
    {
      "0": "arroz_y_granos",
      "1": "pastas"
    }
    """
    if not labels_path.exists():
        raise FileNotFoundError(f"No se encontró labels.json en: {labels_path}")

    with labels_path.open("r", encoding="utf-8") as file:
        labels_dict = json.load(file)

    class_names = [
        labels_dict[str(index)]
        for index in range(len(labels_dict))
    ]

    return class_names


def validate_dataset_dirs(class_names: list[str]) -> None:
    """
    Verifica que existan las carpetas principales y las carpetas de clases.
    """
    required_dirs = [TRAIN_DIR, VALIDATION_DIR, TEST_DIR]

    for directory in required_dirs:
        if not directory.exists():
            raise FileNotFoundError(f"No existe la carpeta requerida: {directory}")

    for split_dir in required_dirs:
        missing_classes = []

        for class_name in class_names:
            class_dir = split_dir / class_name
            if not class_dir.exists():
                missing_classes.append(class_name)

        if missing_classes:
            raise FileNotFoundError(
                f"En {split_dir} faltan estas carpetas de clases: {missing_classes}"
            )


def count_images_by_class(split_dir: Path, class_names: list[str]) -> dict[str, int]:
    """
    Cuenta imágenes por clase en un split específico.
    """
    counts = {}

    valid_extensions = {".jpg", ".jpeg", ".png", ".webp"}

    for class_name in class_names:
        class_dir = split_dir / class_name
        if not class_dir.exists():
            counts[class_name] = 0
            continue

        total = sum(
            1
            for file in class_dir.iterdir()
            if file.is_file() and file.suffix.lower() in valid_extensions
        )

        counts[class_name] = total

    return counts


def create_dataset(directory: Path, class_names: list[str], shuffle: bool) -> tf.data.Dataset:
    """
    Carga imágenes desde carpetas usando image_dataset_from_directory.
    """
    dataset = tf.keras.utils.image_dataset_from_directory(
        directory,
        labels="inferred",
        label_mode="categorical",
        class_names=class_names,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        seed=SEED,
    )

    return dataset


def get_class_weights(train_dataset: tf.data.Dataset, num_classes: int) -> dict[int, float]:
    """
    Calcula pesos por clase para compensar desbalance del dataset.
    """
    y_train = []

    for _, labels in train_dataset:
        label_indices = np.argmax(labels.numpy(), axis=1)
        y_train.extend(label_indices)

    y_train = np.array(y_train)

    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(num_classes),
        y=y_train,
    )

    return {
        index: float(weight)
        for index, weight in enumerate(class_weights)
    }


def build_model(num_classes: int) -> tf.keras.Model:
    """
    Construye el modelo usando MobileNetV2 como extractor de características.
    """
    inputs = tf.keras.Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))

    data_augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.08),
            tf.keras.layers.RandomZoom(0.10),
            tf.keras.layers.RandomContrast(0.10),
        ],
        name="data_augmentation",
    )

    x = data_augmentation(inputs)

    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)

    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3),
        include_top=False,
        weights="imagenet",
        pooling="avg",
    )

    base_model.trainable = False

    x = base_model(x, training=False)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.35)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs, name="despensa_mobilenetv2_classifier")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def plot_training_history(history: tf.keras.callbacks.History, output_path: Path) -> None:
    """
    Guarda una gráfica con accuracy, val_accuracy, loss y val_loss.
    """
    history_dict = history.history

    epochs_range = range(1, len(history_dict["accuracy"]) + 1)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, history_dict["accuracy"], label="Train Accuracy")
    plt.plot(epochs_range, history_dict["val_accuracy"], label="Validation Accuracy")
    plt.xlabel("Época")
    plt.ylabel("Accuracy")
    plt.title("Accuracy durante entrenamiento")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, history_dict["loss"], label="Train Loss")
    plt.plot(epochs_range, history_dict["val_loss"], label="Validation Loss")
    plt.xlabel("Época")
    plt.ylabel("Loss")
    plt.title("Loss durante entrenamiento")
    plt.legend()

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_confusion_matrix(
    matrix: np.ndarray,
    class_names: list[str],
    output_path: Path,
) -> None:
    """
    Guarda la matriz de confusión como imagen.
    """
    plt.figure(figsize=(10, 8))
    plt.imshow(matrix, interpolation="nearest")
    plt.title("Matriz de confusión")
    plt.colorbar()

    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha="right")
    plt.yticks(tick_marks, class_names)

    threshold = matrix.max() / 2 if matrix.max() > 0 else 0

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            plt.text(
                j,
                i,
                str(matrix[i, j]),
                ha="center",
                va="center",
                color="white" if matrix[i, j] > threshold else "black",
            )

    plt.ylabel("Clase real")
    plt.xlabel("Clase predicha")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def evaluate_model(
    model: tf.keras.Model,
    test_dataset: tf.data.Dataset,
    class_names: list[str],
) -> tuple[dict, np.ndarray, dict]:
    """
    Evalúa el modelo en test y genera classification report.
    """
    test_loss, test_accuracy = model.evaluate(test_dataset, verbose=1)

    y_true = []
    y_pred = []

    for images, labels in test_dataset:
        predictions = model.predict(images, verbose=0)

        true_indices = np.argmax(labels.numpy(), axis=1)
        pred_indices = np.argmax(predictions, axis=1)

        y_true.extend(true_indices)
        y_pred.extend(pred_indices)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    cm = confusion_matrix(y_true, y_pred)

    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    metrics = {
        "test_loss": float(test_loss),
        "test_accuracy": float(test_accuracy),
    }

    return metrics, cm, report


def save_training_report(
    output_path: Path,
    class_names: list[str],
    train_counts: dict[str, int],
    validation_counts: dict[str, int],
    test_counts: dict[str, int],
    class_weights: dict[int, float],
    history: tf.keras.callbacks.History,
    test_metrics: dict,
    classification_report_dict: dict,
    confusion_matrix_data: np.ndarray,
) -> None:
    """
    Guarda un reporte completo del entrenamiento en JSON.
    """
    payload = {
        "project": "Sistema Inteligente de Clasificación e Inventariado de Productos de Despensa",
        "model": "MobileNetV2 Transfer Learning",
        "created_at": datetime.now().isoformat(),
        "image_size": IMAGE_SIZE,
        "batch_size": BATCH_SIZE,
        "epochs_configured": EPOCHS,
        "classes": class_names,
        "num_classes": len(class_names),
        "dataset": {
            "train_dir": str(TRAIN_DIR),
            "validation_dir": str(VALIDATION_DIR),
            "test_dir": str(TEST_DIR),
            "train_counts": train_counts,
            "validation_counts": validation_counts,
            "test_counts": test_counts,
        },
        "class_weights": {
            str(key): value
            for key, value in class_weights.items()
        },
        "history": {
            key: [float(value) for value in values]
            for key, values in history.history.items()
        },
        "test_metrics": test_metrics,
        "classification_report": classification_report_dict,
        "confusion_matrix": confusion_matrix_data.tolist(),
        "artifacts": {
            "model_path": str(MODEL_PATH),
            "training_plot": str(ACCURACY_LOSS_PLOT_PATH),
            "confusion_matrix_plot": str(CONFUSION_MATRIX_PATH),
        },
    }

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)


def main() -> None:
    print("\nIniciando entrenamiento del modelo de productos de despensa...\n")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    class_names = load_labels(LABELS_PATH)
    num_classes = len(class_names)

    print("Clases detectadas:")
    for index, class_name in enumerate(class_names):
        print(f"  {index}: {class_name}")

    validate_dataset_dirs(class_names)

    train_counts = count_images_by_class(TRAIN_DIR, class_names)
    validation_counts = count_images_by_class(VALIDATION_DIR, class_names)
    test_counts = count_images_by_class(TEST_DIR, class_names)

    print("\nImágenes por clase en train:")
    for class_name, count in train_counts.items():
        print(f"  {class_name}: {count}")

    print("\nCargando datasets...")

    train_dataset = create_dataset(TRAIN_DIR, class_names, shuffle=True)
    validation_dataset = create_dataset(VALIDATION_DIR, class_names, shuffle=False)
    test_dataset = create_dataset(TEST_DIR, class_names, shuffle=False)

    print("\nCalculando class weights...")
    class_weights = get_class_weights(train_dataset, num_classes)

    for class_index, weight in class_weights.items():
        print(f"  {class_names[class_index]}: {weight:.4f}")

    autotune = tf.data.AUTOTUNE

    train_dataset_prefetch = train_dataset.prefetch(autotune)
    validation_dataset_prefetch = validation_dataset.prefetch(autotune)
    test_dataset_prefetch = test_dataset.prefetch(autotune)

    print("\nConstruyendo modelo MobileNetV2...")
    model = build_model(num_classes)

    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=4,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=MODEL_PATH,
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.3,
            patience=2,
            min_lr=1e-6,
        ),
    ]

    print("\nEntrenando modelo...\n")

    history = model.fit(
        train_dataset_prefetch,
        validation_data=validation_dataset_prefetch,
        epochs=EPOCHS,
        callbacks=callbacks,
        class_weight=class_weights,
    )

    print("\nGuardando modelo final...")
    model.save(MODEL_PATH)

    print("\nEvaluando modelo con test set...")
    test_metrics, cm, classification_report_dict = evaluate_model(
        model,
        test_dataset_prefetch,
        class_names,
    )

    print("\nResultados en test:")
    print(f"  Test loss: {test_metrics['test_loss']:.4f}")
    print(f"  Test accuracy: {test_metrics['test_accuracy']:.4f}")

    print("\nGuardando gráficas y reportes...")

    plot_training_history(history, ACCURACY_LOSS_PLOT_PATH)
    plot_confusion_matrix(cm, class_names, CONFUSION_MATRIX_PATH)

    save_training_report(
        REPORT_PATH,
        class_names,
        train_counts,
        validation_counts,
        test_counts,
        class_weights,
        history,
        test_metrics,
        classification_report_dict,
        cm,
    )

    print("\nEntrenamiento finalizado correctamente.")
    print("\nArchivos generados:")
    print(f"  Modelo: {MODEL_PATH}")
    print(f"  Reporte: {REPORT_PATH}")
    print(f"  Gráfica accuracy/loss: {ACCURACY_LOSS_PLOT_PATH}")
    print(f"  Matriz de confusión: {CONFUSION_MATRIX_PATH}")


if __name__ == "__main__":
    main()