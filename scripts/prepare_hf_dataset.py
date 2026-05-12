"""
Prepare the Hugging Face supermarket products dataset as an ImageFolder dataset.

Run from the project root:
    python scripts/prepare_hf_dataset.py
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import traceback
import random
import re
import shutil
import sys
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Protocol, TypeVar

import pandas as pd
from datasets import Dataset, DatasetDict, Image as HFImage, IterableDatasetDict, load_dataset
from datasets.features import ClassLabel
from PIL import Image as PILImage
from PIL import ImageOps, UnidentifiedImageError
from sklearn.model_selection import train_test_split
from tqdm import tqdm


DATASET_NAME = "valentinafevu/productos-supermercado"

TARGET_CATEGORIES = [
    "arroz_y_granos",
    "pastas",
    "aceites",
    "salsas_y_condimentos",
    "cafe_chocolate",
    "enlatados",
    "azucar_sal",
]

LABELS = {str(index): category for index, category in enumerate(TARGET_CATEGORIES)}

KEYWORD_MAP = {
    "arroz_y_granos": [
        "arroz",
        "frijol",
        "frijoles",
        "lenteja",
        "lentejas",
        "garbanzo",
        "garbanzos",
        "grano",
        "granos",
        "maiz",
        "maíz",
        "quinua",
        "quinoa",
        "cereal",
        "avena",
    ],
    "pastas": [
        "pasta",
        "pastas",
        "spaghetti",
        "espagueti",
        "espaguetis",
        "macarron",
        "macarrones",
        "fideo",
        "fideos",
        "lasagna",
        "lasaña",
    ],
    "aceites": [
        "aceite",
        "aceites",
        "oliva",
        "girasol",
        "canola",
        "vegetal",
    ],
    "salsas_y_condimentos": [
        "salsa",
        "salsas",
        "mayonesa",
        "mostaza",
        "tomate",
        "ketchup",
        "condimento",
        "condimentos",
        "caldo",
        "consome",
        "consomé",
        "vinagre",
        "adobo",
        "sazonador",
        "sal de ajo",
    ],
    "cafe_chocolate": [
        "cafe",
        "café",
        "chocolate",
        "cocoa",
        "chocolisto",
        "milo",
        "instantaneo",
        "instantáneo",
        "bebida achocolatada",
    ],
    "enlatados": [
        "atun",
        "atún",
        "sardina",
        "sardinas",
        "enlatado",
        "enlatados",
        "lata",
        "maiz tierno",
        "maíz tierno",
        "arveja",
        "arvejas",
        "conserva",
        "conservas",
    ],
    "azucar_sal": [
        "azucar",
        "azúcar",
        "sal",
        "panela",
        "endulzante",
        "stevia",
        "edulcorante",
    ],
}

CATEGORY_ALIASES = {
    "arroz_y_granos": [
        "arroz y granos",
        "arroz-y-granos",
        "granos",
        "granos y arroz",
        "legumbres",
        "cereales",
        "avena",
    ],
    "pastas": [
        "pastas",
        "pasta",
        "pastas alimenticias",
        "fideos",
    ],
    "aceites": [
        "aceites",
        "aceite",
        "aceites y vinagres",
    ],
    "salsas_y_condimentos": [
        "salsas",
        "salsas y condimentos",
        "condimentos",
        "caldos",
        "aderezos",
        "mayonesa",
        "mostaza",
    ],
    "cafe_chocolate": [
        "cafe",
        "café",
        "chocolate",
        "cafe y chocolate",
        "café y chocolate",
        "bebidas calientes",
    ],
    "enlatados": [
        "enlatados",
        "conservas",
        "atun",
        "atún",
        "sardinas",
        "latas",
    ],
    "azucar_sal": [
        "azucar",
        "azúcar",
        "sal",
        "panela",
        "azucar y sal",
        "azúcar y sal",
        "endulzantes",
    ],
}

IMAGE_COLUMN_CANDIDATES = [
    "image",
    "imagen",
    "img",
    "photo",
    "picture",
    "file",
    "filepath",
    "file_path",
    "path",
    "image_path",
    "ruta",
    "ruta_imagen",
]

NAME_COLUMN_CANDIDATES = [
    "producto",
    "nombre",
    "name",
    "product",
    "product_name",
    "nombre_producto",
    "title",
    "descripcion",
    "description",
    "texto",
    "text",
]

CATEGORY_COLUMN_CANDIDATES = [
    "main_category",
    "categoria_principal",
    "categoria_general",
    "primary_category",
    "product_category",
    "categoria",
    "category",
    "label",
    "labels",
    "clase",
    "class",
    "etiqueta",
    "tipo",
    "grupo",
]


@dataclass(frozen=True)
class ProductRecord:
    source_split: str
    index: int
    name: str
    source_category: str
    category: str
    classification_reason: str


@dataclass(frozen=True)
class SavedImageRecord(ProductRecord):
    path: Path


class CategorizedRecord(Protocol):
    category: str


RecordT = TypeVar("RecordT", bound=CategorizedRecord)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download, inspect and prepare a Hugging Face image dataset."
    )
    parser.add_argument("--dataset-name", default=DATASET_NAME)
    parser.add_argument("--output-dir", default="datasets/processed")
    parser.add_argument("--image-column", default=None)
    parser.add_argument("--name-column", default=None)
    parser.add_argument("--category-column", default=None)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--image-format", choices=["jpg", "png"], default="jpg")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument(
        "--streaming",
        dest="streaming",
        action="store_true",
        default=True,
        help="Read the dataset in streaming mode. Recommended for this dataset.",
    )
    parser.add_argument(
        "--no-streaming",
        dest="streaming",
        action="store_false",
        help="Download/prepare the full dataset before processing.",
    )
    parser.add_argument(
        "--max-records",
        type=int,
        default=None,
        help="Optional limit for quick tests. Processes all records by default.",
    )
    parser.add_argument(
        "--save-otros",
        dest="save_otros",
        action="store_true",
        default=True,
        help="Save unclassified images in datasets/processed/otros.",
    )
    parser.add_argument(
        "--no-save-otros",
        dest="save_otros",
        action="store_false",
        help="Do not save unclassified images; only count them in the report.",
    )
    parser.add_argument(
    "--clean-output",
    action="store_true",
    help="Delete the output directory before processing. Useful after failed runs.",
    )   
    return parser.parse_args()


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).lower().strip()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(char for char in text if not unicodedata.combining(char))
    text = re.sub(r"[_\-./,;:(){}\[\]+*|!?\"']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def keyword_in_text(keyword: str, text: str) -> bool:
    keyword = normalize_text(keyword)
    if not keyword:
        return False
    pattern = r"(?<!\w)" + re.escape(keyword).replace(r"\ ", r"\s+") + r"(?!\w)"
    return re.search(pattern, text) is not None


def slugify(value: str, fallback: str = "producto") -> str:
    slug = normalize_text(value)
    slug = re.sub(r"[^a-z0-9]+", "-", slug).strip("-")
    return (slug or fallback)[:60]


def load_hf_dataset(
    dataset_name: str,
    trust_remote_code: bool,
    streaming: bool,
) -> Dataset | DatasetDict | IterableDatasetDict:
    try:
        return load_dataset(
            dataset_name,
            trust_remote_code=trust_remote_code,
            streaming=streaming,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(f"Dataset no encontrado: {dataset_name}") from exc
    except Exception as exc:
        raise RuntimeError(
            f"No se pudo descargar o cargar el dataset '{dataset_name}'. "
            "Revisa conexion, nombre del dataset y permisos de Hugging Face.\n"
            f"Detalle tecnico: {type(exc).__name__}: {exc}"
        ) from exc


def as_split_dict(dataset: Dataset | DatasetDict | IterableDatasetDict) -> dict[str, Any]:
    if isinstance(dataset, (DatasetDict, IterableDatasetDict)):
        return {split_name: split_dataset for split_name, split_dataset in dataset.items()}
    return {"all": dataset}


def cast_image_column_decode_false(
    splits: dict[str, Any],
    image_column: str | None,
) -> dict[str, Any]:
    if not image_column:
        return splits

    prepared: dict[str, Any] = {}

    for split_name, dataset in splits.items():
        try:
            if hasattr(dataset, "cast_column"):
                prepared[split_name] = dataset.cast_column(
                    image_column,
                    HFImage(decode=False),
                )
            else:
                prepared[split_name] = dataset
        except Exception:
            prepared[split_name] = dataset

    return prepared


def get_all_columns(splits: dict[str, Dataset]) -> list[str]:
    columns: list[str] = []
    for dataset in splits.values():
        for column in dataset.column_names:
            if column not in columns:
                columns.append(column)
    return columns


def detect_column_by_name(columns: Iterable[str], candidates: list[str]) -> str | None:
    normalized = {normalize_text(column): column for column in columns}
    for candidate in candidates:
        found = normalized.get(normalize_text(candidate))
        if found:
            return found
    for column in columns:
        column_text = normalize_text(column)
        if any(normalize_text(candidate) in column_text for candidate in candidates):
            return column
    return None


def detect_image_column(splits: dict[str, Dataset], override: str | None) -> str | None:
    columns = get_all_columns(splits)
    if override:
        return override if override in columns else None

    for dataset in splits.values():
        for column, feature in dataset.features.items():
            if isinstance(feature, HFImage):
                return column

    return detect_column_by_name(columns, IMAGE_COLUMN_CANDIDATES)


def detect_name_column(splits: dict[str, Dataset], override: str | None) -> str | None:
    columns = get_all_columns(splits)
    if override:
        return override if override in columns else None
    return detect_column_by_name(columns, NAME_COLUMN_CANDIDATES)


def detect_category_column(splits: dict[str, Dataset], override: str | None) -> str | None:
    columns = get_all_columns(splits)
    if override:
        return override if override in columns else None

    # The source dataset includes both "supermarket_category" and
    # "main_category"; the latter is the useful training label.
    for preferred in ["main_category", "categoria_principal", "categoria_general"]:
        for column in columns:
            if normalize_text(column) == normalize_text(preferred):
                return column

    return detect_column_by_name(columns, CATEGORY_COLUMN_CANDIDATES)


def cast_image_columns_for_manual_validation(
    splits: dict[str, Dataset], image_column: str
) -> dict[str, Dataset]:
    prepared: dict[str, Dataset] = {}
    for split_name, dataset in splits.items():
        if image_column not in dataset.column_names:
            prepared[split_name] = dataset
            continue

        feature = dataset.features.get(image_column)
        if isinstance(feature, HFImage):
            prepared[split_name] = dataset.cast_column(image_column, HFImage(decode=False))
        else:
            prepared[split_name] = dataset
    return prepared


def decode_class_label(dataset: Dataset, column: str | None, value: Any) -> str:
    if column is None or value is None:
        return ""
    feature = dataset.features.get(column)
    if isinstance(feature, ClassLabel):
        try:
            return str(feature.int2str(int(value)))
        except Exception:
            return str(value)
    if isinstance(value, list):
        return " ".join(str(item) for item in value)
    return str(value)


def image_value_present(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, PILImage.Image):
        return True
    if isinstance(value, dict):
        return bool(value.get("path") or value.get("bytes"))
    if isinstance(value, (str, Path)):
        return bool(str(value).strip())
    return True


def describe_image_value(value: Any) -> str:
    if value is None:
        return "sin imagen"
    if isinstance(value, PILImage.Image):
        return f"PIL.Image(mode={value.mode}, size={value.size})"
    if isinstance(value, dict):
        parts = []
        if value.get("path"):
            parts.append(f"path={value.get('path')}")
        if value.get("bytes"):
            parts.append(f"bytes={len(value.get('bytes'))}")
        return "Image(" + ", ".join(parts or ["vacio"]) + ")"
    return str(value)[:160]


def sanitize_example(
    item: dict[str, Any],
    columns: list[str],
    image_column: str | None,
) -> dict[str, Any]:
    example: dict[str, Any] = {}
    for column in columns:
        value = item.get(column)
        if column == image_column:
            example[column] = describe_image_value(value)
        else:
            text = str(value)
            example[column] = text[:300] + ("..." if len(text) > 300 else "")
    return example


def map_existing_category(value: str) -> str | None:
    text = normalize_text(value)
    if not text:
        return None

    # Mapeo directo de posibles valores reales de subcategory.
    direct_map = {
        "arroz y granos": "arroz_y_granos",
        "arroz-y-granos": "arroz_y_granos",
        "granos": "arroz_y_granos",
        "legumbres": "arroz_y_granos",
        "cereales": "arroz_y_granos",
        "avena": "arroz_y_granos",

        "pastas": "pastas",
        "pasta": "pastas",
        "fideos": "pastas",

        "aceites": "aceites",
        "aceite": "aceites",
        "aceites y vinagres": "aceites",

        "salsas": "salsas_y_condimentos",
        "condimentos": "salsas_y_condimentos",
        "salsas y condimentos": "salsas_y_condimentos",
        "caldos": "salsas_y_condimentos",
        "aderezos": "salsas_y_condimentos",

        "cafe": "cafe_chocolate",
        "café": "cafe_chocolate",
        "chocolate": "cafe_chocolate",
        "cafe y chocolate": "cafe_chocolate",
        "café y chocolate": "cafe_chocolate",

        "enlatados": "enlatados",
        "conservas": "enlatados",
        "atun": "enlatados",
        "atún": "enlatados",
        "sardinas": "enlatados",

        "azucar": "azucar_sal",
        "azúcar": "azucar_sal",
        "sal": "azucar_sal",
        "panela": "azucar_sal",
        "azucar y sal": "azucar_sal",
        "azúcar y sal": "azucar_sal",
        "endulzantes": "azucar_sal",
    }

    if text in direct_map:
        return direct_map[text]

    for source, target in direct_map.items():
        if source in text:
            return target

    for category, aliases in CATEGORY_ALIASES.items():
        if any(keyword_in_text(alias, text) for alias in aliases):
            return category

    for category in TARGET_CATEGORIES:
        normalized_category = normalize_text(category.replace("_", " "))
        if normalized_category in text:
            return category

    return None


def infer_category(product_name: str, source_category: str = "") -> tuple[str | None, str]:
    mapped = map_existing_category(source_category)
    if mapped:
        return mapped, f"categoria_existente:{source_category}"

    text = normalize_text(f"{product_name} {source_category}")
    for category, keywords in KEYWORD_MAP.items():
        for keyword in keywords:
            if keyword_in_text(keyword, text):
                return category, f"palabra_clave:{keyword}"

    return None, "sin_coincidencia_confiable"

def clean_output_dir_safely(output_dir: Path) -> None:
    resolved = output_dir.resolve()
    dangerous_paths = {
        Path(".").resolve(),
        Path.home().resolve(),
        Path(resolved.anchor).resolve(),
    }

    if resolved in dangerous_paths:
        raise RuntimeError(f"Ruta peligrosa para borrar: {resolved}")

    if len(resolved.parts) < 3:
        raise RuntimeError(f"Ruta demasiado general para borrar: {resolved}")

    if resolved.exists():
        shutil.rmtree(resolved)


def create_output_dirs(output_dir: Path, save_otros: bool) -> None:
    for split in ["train", "validation", "test"]:
        for category in TARGET_CATEGORIES:
            (output_dir / split / category).mkdir(parents=True, exist_ok=True)
    if save_otros:
        (output_dir / "otros").mkdir(parents=True, exist_ok=True)


def inspect_and_classify_records(
    splits: dict[str, Dataset],
    image_column: str,
    name_column: str | None,
    category_column: str | None,
) -> tuple[
    list[ProductRecord],
    list[ProductRecord],
    Counter,
    Counter,
    list[dict[str, Any]],
    dict[str, Any],
]:
    classified: list[ProductRecord] = []
    unclassified: list[ProductRecord] = []
    discarded_reasons: Counter = Counter()
    original_category_counts: Counter = Counter()
    examples: list[dict[str, Any]] = []
    records_without_name = 0
    split_lengths = {split_name: len(dataset) for split_name, dataset in splits.items()}
    all_columns = get_all_columns(splits)

    for split_name, dataset in splits.items():
        progress = tqdm(
            range(len(dataset)),
            desc=f"Inspeccionando {split_name}",
            unit="reg",
        )
        for index in progress:
            try:
                item = dataset[index]
            except Exception:
                discarded_reasons["registro_ilegible"] += 1
                continue

            if len(examples) < 5:
                examples.append(sanitize_example(item, all_columns, image_column))

            if image_column not in item or not image_value_present(item.get(image_column)):
                discarded_reasons["sin_imagen"] += 1
                continue

            product_name = str(item.get(name_column, "")).strip() if name_column else ""
            source_category = decode_class_label(
                dataset, category_column, item.get(category_column) if category_column else None
            ).strip()
            if source_category:
                original_category_counts[source_category] += 1

            category, reason = infer_category(product_name, source_category)
            record = ProductRecord(
                source_split=split_name,
                index=index,
                name=product_name,
                source_category=source_category,
                category=category or "otros",
                classification_reason=reason,
            )

            if not product_name:
                records_without_name += 1

            if category:
                classified.append(record)
            else:
                unclassified.append(record)

    inspection = {
        "total_records": sum(split_lengths.values()),
        "original_splits": split_lengths,
        "columns": all_columns,
        "image_column": image_column,
        "name_column": name_column,
        "category_column": category_column,
        "contains_images": image_column is not None,
        "contains_product_names": name_column is not None,
        "contains_categories": category_column is not None,
        "examples": examples,
        "possible_categories": list(original_category_counts.keys())[:100],
        "original_category_counts": dict(original_category_counts.most_common()),
        "records_without_name": records_without_name,
    }

    return (
        classified,
        unclassified,
        discarded_reasons,
        original_category_counts,
        examples,
        inspection,
    )


def split_records_with_sklearn(
    records: list[RecordT], seed: int
) -> dict[str, list[RecordT]]:
    categories = [record.category for record in records]
    train_records, temp_records = train_test_split(
        records,
        train_size=0.70,
        random_state=seed,
        shuffle=True,
        stratify=categories,
    )
    temp_categories = [record.category for record in temp_records]
    validation_records, test_records = train_test_split(
        temp_records,
        train_size=0.50,
        random_state=seed,
        shuffle=True,
        stratify=temp_categories,
    )
    return {
        "train": list(train_records),
        "validation": list(validation_records),
        "test": list(test_records),
    }


def compute_split_counts(total: int) -> tuple[int, int, int]:
    if total <= 0:
        return 0, 0, 0
    if total == 1:
        return 1, 0, 0
    if total == 2:
        return 1, 0, 1

    train_count = max(1, int(round(total * 0.70)))
    validation_count = max(1, int(round(total * 0.15)))
    test_count = total - train_count - validation_count

    while test_count < 1:
        if train_count >= validation_count and train_count > 1:
            train_count -= 1
        elif validation_count > 1:
            validation_count -= 1
        else:
            break
        test_count = total - train_count - validation_count

    while train_count + validation_count + test_count > total:
        train_count -= 1
    while train_count + validation_count + test_count < total:
        train_count += 1

    return train_count, validation_count, test_count


def split_records_per_category(
    records: list[RecordT], seed: int
) -> dict[str, list[RecordT]]:
    rng = random.Random(seed)
    by_category: dict[str, list[ProductRecord]] = defaultdict(list)
    for record in records:
        by_category[record.category].append(record)

    splits = {"train": [], "validation": [], "test": []}
    for category_records in by_category.values():
        rng.shuffle(category_records)
        train_count, validation_count, test_count = compute_split_counts(len(category_records))
        splits["train"].extend(category_records[:train_count])
        splits["validation"].extend(
            category_records[train_count : train_count + validation_count]
        )
        splits["test"].extend(
            category_records[
                train_count + validation_count : train_count + validation_count + test_count
            ]
        )

    for split_records in splits.values():
        rng.shuffle(split_records)
    return splits


def split_records(
    records: list[RecordT], seed: int
) -> tuple[dict[str, list[RecordT]], str]:
    if not records:
        return {"train": [], "validation": [], "test": []}, "sin_registros"

    category_counts = Counter(record.category for record in records)
    try:
        if len(category_counts) > 1 and min(category_counts.values()) >= 5:
            return split_records_with_sklearn(records, seed), "sklearn_estratificado"
    except ValueError:
        pass

    return split_records_per_category(records, seed), "por_categoria_con_respaldo"


def open_and_validate_image(value: Any, image_size: int | None) -> PILImage.Image:
    try:
        if isinstance(value, PILImage.Image):
            image = value.copy()
        elif isinstance(value, dict):
            if value.get("bytes"):
                image = PILImage.open(io.BytesIO(value["bytes"]))
            elif value.get("path"):
                image = PILImage.open(value["path"])
            else:
                raise ValueError("registro sin path ni bytes de imagen")
        elif isinstance(value, (str, Path)):
            image = PILImage.open(value)
        else:
            raise ValueError(f"tipo de imagen no soportado: {type(value)!r}")

        image = ImageOps.exif_transpose(image)
        image.load()
        image = image.convert("RGB")
        if image_size and image_size > 0:
            image = image.resize((image_size, image_size), PILImage.Resampling.LANCZOS)
        return image
    except (OSError, ValueError, UnidentifiedImageError) as exc:
        raise ValueError(f"imagen danada o no valida: {exc}") from exc


def unique_filename(record: ProductRecord, extension: str) -> str:
    key = f"{record.source_split}:{record.index}:{record.name}:{record.category}"
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]
    name_part = slugify(record.name or record.source_category or record.category)
    return f"{record.category}_{digest}_{name_part}.{extension}"


def save_record_image(
    record: ProductRecord,
    source_dataset: Dataset,
    image_column: str,
    destination_dir: Path,
    image_size: int | None,
    image_format: str,
) -> Path:
    item = source_dataset[record.index]
    image_value = item.get(image_column)
    return save_image_value(record, image_value, destination_dir, image_size, image_format)


def save_image_value(
    record: ProductRecord,
    image_value: Any,
    destination_dir: Path,
    image_size: int | None,
    image_format: str,
) -> Path:
    destination_dir.mkdir(parents=True, exist_ok=True)
    image = open_and_validate_image(image_value, image_size)

    extension = "jpg" if image_format == "jpg" else "png"
    destination = destination_dir / unique_filename(record, extension)
    counter = 1
    while destination.exists():
        destination = destination.with_name(
            f"{destination.stem}_{counter}{destination.suffix}"
        )
        counter += 1

    if image_format == "jpg":
        image.save(destination, format="JPEG", quality=95, optimize=True)
    else:
        image.save(destination, format="PNG", optimize=True)
    return destination


def inspect_classify_and_stage_streaming(
    splits: dict[str, Any],
    args: argparse.Namespace,
    output_dir: Path,
    image_size: int | None,
) -> tuple[list[SavedImageRecord], list[ProductRecord], Counter, dict[str, Any], int]:
    image_column = args.image_column
    name_column = args.name_column
    category_column = args.category_column
    staging_dir = output_dir / "_staging"

    saved_records: list[SavedImageRecord] = []
    unclassified_records: list[ProductRecord] = []
    discarded_reasons: Counter = Counter()
    original_category_counts: Counter = Counter()
    examples: list[dict[str, Any]] = []
    all_columns: list[str] = []
    split_lengths: Counter = Counter()
    records_without_name = 0
    total_seen = 0
    otros_saved = 0

    for split_name, dataset in splits.items():
        iterator = iter(dataset)
        progress = tqdm(
            desc=f"Procesando streaming {split_name}",
            unit="reg",
        )
        index = -1

        while True:
            if args.max_records is not None and total_seen >= args.max_records:
                break

            try:
                index += 1
                item = next(iterator)
            except StopIteration:
                break
            except Exception:
                discarded_reasons["registro_ilegible_o_imagen_no_decodificable"] += 1
                progress.update(1)
                continue

            progress.update(1)

            total_seen += 1
            split_lengths[split_name] += 1
            for column in item.keys():
                if column not in all_columns:
                    all_columns.append(column)

            if image_column is None:
                image_column = detect_column_by_name(all_columns, IMAGE_COLUMN_CANDIDATES)
            if name_column is None:
                name_column = detect_column_by_name(all_columns, NAME_COLUMN_CANDIDATES)
            if category_column is None:
                category_column = detect_column_by_name(
                    all_columns,
                    CATEGORY_COLUMN_CANDIDATES,
                )

            if image_column is None:
                discarded_reasons["sin_columna_imagen"] += 1
                continue

            if len(examples) < 5:
                try:
                    examples.append(sanitize_example(item, all_columns, image_column))
                except Exception:
                    pass

            image_value = item.get(image_column)
            if not image_value_present(image_value):
                discarded_reasons["sin_imagen"] += 1
                continue

            product_name = str(item.get(name_column, "")).strip() if name_column else ""
            source_category = (
                str(item.get(category_column, "")).strip() if category_column else ""
            )
            if source_category:
                original_category_counts[source_category] += 1
            if not product_name:
                records_without_name += 1

            

            category, reason = infer_category(product_name, source_category)
            record = ProductRecord(
                source_split=split_name,
                index=index,
                name=product_name,
                source_category=source_category,
                category=category or "otros",
                classification_reason=reason,
            )

            try:
                if category:
                    destination = save_image_value(
                        record,
                        image_value,
                        staging_dir / category,
                        image_size,
                        args.image_format,
                    )
                    saved_records.append(
                        SavedImageRecord(
                            source_split=record.source_split,
                            index=record.index,
                            name=record.name,
                            source_category=record.source_category,
                            category=record.category,
                            classification_reason=record.classification_reason,
                            path=destination,
                        )
                    )
                else:
                    unclassified_records.append(record)
                    if args.save_otros:
                        save_image_value(
                            record,
                            image_value,
                            output_dir / "otros",
                            image_size,
                            args.image_format,
                        )
                        otros_saved += 1
            except Exception:
                if category:
                    discarded_reasons["imagen_danada_o_error_guardado"] += 1
                else:
                    discarded_reasons["otros_imagen_danada_o_error_guardado"] += 1

        if args.max_records is not None and total_seen >= args.max_records:
            break

    inspection = {
        "total_records": total_seen,
        "original_splits": dict(split_lengths),
        "columns": all_columns,
        "image_column": image_column,
        "name_column": name_column,
        "category_column": category_column,
        "contains_images": image_column is not None,
        "contains_product_names": name_column is not None,
        "contains_categories": category_column is not None,
        "examples": examples,
        "possible_categories": list(original_category_counts.keys())[:100],
        "original_category_counts": dict(original_category_counts.most_common()),
        "records_without_name": records_without_name,
        "streaming": True,
    }

    return saved_records, unclassified_records, discarded_reasons, inspection, otros_saved


def destination_with_unique_name(destination: Path) -> Path:
    if not destination.exists():
        return destination

    counter = 1
    while True:
        candidate = destination.with_name(
            f"{destination.stem}_{counter}{destination.suffix}"
        )
        if not candidate.exists():
            return candidate
        counter += 1


def move_staged_images_to_splits(
    splits: dict[str, list[SavedImageRecord]],
    output_dir: Path,
) -> tuple[Counter, dict[str, Counter], Counter]:
    images_by_category: Counter = Counter()
    images_by_split: Counter = Counter()
    split_category_counts: dict[str, Counter] = {
        split_name: Counter() for split_name in ["train", "validation", "test"]
    }

    for split_name, records in splits.items():
        for record in tqdm(records, desc=f"Moviendo {split_name}", unit="img"):
            destination_dir = output_dir / split_name / record.category
            destination_dir.mkdir(parents=True, exist_ok=True)
            destination = destination_with_unique_name(destination_dir / record.path.name)
            record.path.replace(destination)
            images_by_category[record.category] += 1
            images_by_split[split_name] += 1
            split_category_counts[split_name][record.category] += 1

    return images_by_category, split_category_counts, images_by_split


def remove_staging_dir(output_dir: Path) -> None:
    staging_dir = (output_dir / "_staging").resolve()
    resolved_output = output_dir.resolve()
    if staging_dir.exists() and str(staging_dir).startswith(str(resolved_output)):
        shutil.rmtree(staging_dir)


def save_split_images(
    splits: dict[str, list[ProductRecord]],
    source_splits: dict[str, Dataset],
    image_column: str,
    output_dir: Path,
    image_size: int | None,
    image_format: str,
) -> tuple[Counter, dict[str, Counter], Counter, Counter]:
    images_by_category: Counter = Counter()
    images_by_split: Counter = Counter()
    discarded_reasons: Counter = Counter()
    split_category_counts: dict[str, Counter] = {
        split_name: Counter() for split_name in ["train", "validation", "test"]
    }

    for split_name, records in splits.items():
        for record in tqdm(records, desc=f"Guardando {split_name}", unit="img"):
            try:
                destination_dir = output_dir / split_name / record.category
                save_record_image(
                    record,
                    source_splits[record.source_split],
                    image_column,
                    destination_dir,
                    image_size,
                    image_format,
                )
                images_by_category[record.category] += 1
                images_by_split[split_name] += 1
                split_category_counts[split_name][record.category] += 1
            except Exception:
                discarded_reasons["imagen_danada_o_error_guardado"] += 1

    return images_by_category, split_category_counts, images_by_split, discarded_reasons


def save_otros_images(
    records: list[ProductRecord],
    source_splits: dict[str, Dataset],
    image_column: str,
    output_dir: Path,
    image_size: int | None,
    image_format: str,
    enabled: bool,
) -> tuple[int, Counter]:
    if not enabled:
        return 0, Counter({"otros_no_guardados": len(records)})

    saved = 0
    discarded_reasons: Counter = Counter()
    destination_dir = output_dir / "otros"
    for record in tqdm(records, desc="Guardando otros", unit="img"):
        try:
            save_record_image(
                record,
                source_splits[record.source_split],
                image_column,
                destination_dir,
                image_size,
                image_format,
            )
            saved += 1
        except Exception:
            discarded_reasons["otros_imagen_danada_o_error_guardado"] += 1
    return saved, discarded_reasons


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def build_report_payload(
    args: argparse.Namespace,
    inspection: dict[str, Any],
    split_method: str,
    images_by_category: Counter,
    split_category_counts: dict[str, Counter],
    images_by_split: Counter,
    discarded_reasons: Counter,
    total_otros: int,
    otros_saved: int,
    unclassified: list[ProductRecord],
) -> dict[str, Any]:
    total_processed = sum(images_by_category.values())
    empty_categories = [
        category for category in TARGET_CATEGORIES if images_by_category.get(category, 0) == 0
    ]
    category_summary = pd.DataFrame(
        [
            {"category": category, "images": int(images_by_category.get(category, 0))}
            for category in TARGET_CATEGORIES
        ]
    ).to_dict(orient="records")
    split_summary = pd.DataFrame(
        [
            {"split": split, "images": int(images_by_split.get(split, 0))}
            for split in ["train", "validation", "test"]
        ]
    ).to_dict(orient="records")

    return {
        "dataset_original": args.dataset_name,
        "processing_date": datetime.now(timezone.utc).isoformat(),
        "output_dir": str(Path(args.output_dir)),
        "image_size": args.image_size if args.image_size > 0 else None,
        "image_format": args.image_format,
        "split_ratio": {"train": 0.70, "validation": 0.15, "test": 0.15},
        "split_method": split_method,
        "labels": LABELS,
        "initial_inspection": inspection,
        "total_images_processed": total_processed,
        "total_images_discarded": sum(discarded_reasons.values()),
        "total_images_otros": total_otros,
        "total_images_otros_saved": otros_saved,
        "images_by_category": {
            category: int(images_by_category.get(category, 0))
            for category in TARGET_CATEGORIES
        },
        "category_summary": category_summary,
        "images_by_split": {
            split: int(images_by_split.get(split, 0))
            for split in ["train", "validation", "test"]
        },
        "split_summary": split_summary,
        "split_category_counts": {
            split: {
                category: int(counts.get(category, 0))
                for category in TARGET_CATEGORIES
            }
            for split, counts in split_category_counts.items()
        },
        "discarded_by_reason": dict(discarded_reasons),
        "empty_categories": empty_categories,
        "classification_criteria": KEYWORD_MAP,
        "unclassified_examples": [
            {
                "source_split": record.source_split,
                "index": record.index,
                "name": record.name,
                "source_category": record.source_category,
                "reason": record.classification_reason,
            }
            for record in unclassified[:100]
        ],
    }


def write_markdown_report(path: Path, payload: dict[str, Any]) -> None:
    lines: list[str] = []
    lines.append("# Reporte de preparacion del dataset")
    lines.append("")
    lines.append(f"- Dataset usado: `{payload['dataset_original']}`")
    lines.append(f"- Fecha de procesamiento: `{payload['processing_date']}`")
    lines.append(f"- Metodo de division: `{payload['split_method']}`")
    lines.append("")
    lines.append("## Inspeccion inicial")
    inspection = payload["initial_inspection"]
    lines.append(f"- Total de registros: {inspection['total_records']}")
    lines.append(f"- Splits originales: `{inspection['original_splits']}`")
    lines.append(f"- Columnas encontradas: `{inspection['columns']}`")
    lines.append(f"- Columna de imagen: `{inspection['image_column']}`")
    lines.append(f"- Columna de nombre/producto: `{inspection['name_column']}`")
    lines.append(f"- Columna de categoria/etiqueta: `{inspection['category_column']}`")
    lines.append("")
    lines.append("## Categorias finales")
    for category, count in payload["images_by_category"].items():
        lines.append(f"- {category}: {count}")
    lines.append("")
    lines.append("## Division final")
    for split, count in payload["images_by_split"].items():
        lines.append(f"- {split}: {count}")
    lines.append("")
    lines.append("## Criterios de clasificacion")
    lines.append(
        "Primero se uso la categoria original del dataset cuando podia mapearse "
        "a una de las cinco categorias objetivo. Cuando no habia una categoria "
        "clara, se infirio la categoria usando palabras clave normalizadas "
        "del nombre del producto."
    )
    lines.append("")
    for category, keywords in payload["classification_criteria"].items():
        lines.append(f"- {category}: {', '.join(keywords)}")
    lines.append("")
    lines.append("## Productos no clasificados")
    lines.append(f"- Total enviados a otros: {payload['total_images_otros']}")
    lines.append(f"- Imagenes guardadas en otros: {payload['total_images_otros_saved']}")
    if payload["unclassified_examples"]:
        lines.append("")
        lines.append("Ejemplos:")
        for example in payload["unclassified_examples"][:30]:
            name = example["name"] or "(sin nombre)"
            source_category = example["source_category"] or "(sin categoria)"
            lines.append(
                f"- split={example['source_split']}, index={example['index']}, "
                f"nombre={name}, categoria_original={source_category}"
            )
    lines.append("")
    lines.append("## Descartes y advertencias")
    lines.append(f"- Total de imagenes descartadas: {payload['total_images_discarded']}")
    for reason, count in payload["discarded_by_reason"].items():
        lines.append(f"- {reason}: {count}")
    if payload["empty_categories"]:
        lines.append(f"- Categorias vacias: {', '.join(payload['empty_categories'])}")
    else:
        lines.append("- No quedaron categorias vacias.")
    lines.append("")
    lines.append("## Primeros ejemplos inspeccionados")
    for index, example in enumerate(inspection["examples"], start=1):
        lines.append(f"### Ejemplo {index}")
        lines.append("```json")
        lines.append(json.dumps(example, indent=2, ensure_ascii=False))
        lines.append("```")
    lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def print_summary(
    report: dict[str, Any],
    labels_path: Path,
    json_report_path: Path,
    md_report_path: Path,
) -> None:
    print("\nDataset preparado correctamente.\n")
    print("Resumen:")
    print(f"- Total imagenes procesadas: {report['total_images_processed']}")
    print(f"- Total imagenes descartadas: {report['total_images_discarded']}")
    print("- Categorias:")
    for category, count in report["images_by_category"].items():
        print(f"  - {category}: {count}")
    print(f"- Train: {report['images_by_split']['train']}")
    print(f"- Validation: {report['images_by_split']['validation']}")
    print(f"- Test: {report['images_by_split']['test']}")
    print(f"- labels.json generado: {labels_path}")
    print(f"- dataset_report.json generado: {json_report_path}")
    print(f"- dataset_report.md generado: {md_report_path}")


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    image_size = args.image_size if args.image_size > 0 else None

    try:
        if args.clean_output:
            clean_output_dir_safely(output_dir)

        dataset = load_hf_dataset(
            args.dataset_name,
            args.trust_remote_code,
            args.streaming,
        )
        raw_splits = as_split_dict(dataset)

        # if args.streaming:
        #    raw_splits = cast_image_column_decode_false(raw_splits, args.image_column)

        create_output_dirs(output_dir, args.save_otros)

        if args.streaming:
            (
                saved_records,
                unclassified_records,
                discarded_reasons,
                inspection,
                otros_saved,
            ) = inspect_classify_and_stage_streaming(
                raw_splits,
                args,
                output_dir,
                image_size,
            )

            if not saved_records:
                raise RuntimeError(
                    "No se guardo ninguna imagen valida. "
                    "Las categorias se leyeron correctamente, pero las imagenes no pudieron abrirse o guardarse. "
                    "Revisa si la columna image esta llegando como PIL.Image, bytes o path relativo."
                )

            record_splits, split_method = split_records(saved_records, args.seed)
            (
                images_by_category,
                split_category_counts,
                images_by_split,
            ) = move_staged_images_to_splits(record_splits, output_dir)
            remove_staging_dir(output_dir)

            labels_path = output_dir / "labels.json"
            json_report_path = output_dir / "dataset_report.json"
            md_report_path = output_dir / "dataset_report.md"

            write_json(labels_path, LABELS)
            report = build_report_payload(
                args,
                inspection,
                f"{split_method}_streaming",
                images_by_category,
                split_category_counts,
                images_by_split,
                discarded_reasons,
                total_otros=len(unclassified_records),
                otros_saved=otros_saved,
                unclassified=unclassified_records,
            )
            write_json(json_report_path, report)
            write_markdown_report(md_report_path, report)
            print_summary(report, labels_path, json_report_path, md_report_path)
            return 0

        image_column = detect_image_column(raw_splits, args.image_column)
        if image_column is None:
            raise RuntimeError(
                "No se encontro una columna de imagen. Usa --image-column para indicarla."
            )

        name_column = detect_name_column(raw_splits, args.name_column)
        category_column = detect_category_column(raw_splits, args.category_column)
        source_splits = cast_image_columns_for_manual_validation(raw_splits, image_column)

        (
            classified_records,
            unclassified_records,
            discarded_reasons,
            _original_category_counts,
            _examples,
            inspection,
        ) = inspect_and_classify_records(
            source_splits,
            image_column,
            name_column,
            category_column,
        )

        if not classified_records:
            raise RuntimeError(
                "No se clasifico ningun registro en las categorias objetivo. "
                "Revisa columnas, nombres de productos o palabras clave."
            )

        record_splits, split_method = split_records(classified_records, args.seed)

        (
            images_by_category,
            split_category_counts,
            images_by_split,
            save_discarded_reasons,
        ) = save_split_images(
            record_splits,
            source_splits,
            image_column,
            output_dir,
            image_size,
            args.image_format,
        )
        discarded_reasons.update(save_discarded_reasons)

        otros_saved, otros_discarded = save_otros_images(
            unclassified_records,
            source_splits,
            image_column,
            output_dir,
            image_size,
            args.image_format,
            args.save_otros,
        )
        discarded_reasons.update(otros_discarded)

        labels_path = output_dir / "labels.json"
        json_report_path = output_dir / "dataset_report.json"
        md_report_path = output_dir / "dataset_report.md"

        write_json(labels_path, LABELS)
        report = build_report_payload(
            args,
            inspection,
            split_method,
            images_by_category,
            split_category_counts,
            images_by_split,
            discarded_reasons,
            total_otros=len(unclassified_records),
            otros_saved=otros_saved,
            unclassified=unclassified_records,
        )
        write_json(json_report_path, report)
        write_markdown_report(md_report_path, report)
        print_summary(report, labels_path, json_report_path, md_report_path)
        return 0

    except KeyboardInterrupt:
        print("\nProceso interrumpido por el usuario.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"\nError preparando el dataset: {exc}", file=sys.stderr)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
