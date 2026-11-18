"""ASL Alphabet keşfi, sızıntı azaltıcı bölme ve batch veri okuma araçları."""

from __future__ import annotations

import json
import queue
import re
import threading
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np

from config import settings
from recognition.preprocessing import PreprocessConfig, preprocess_image

SUPPORTED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
_NUMBER_PATTERN = re.compile(r"(\d+)$")


@dataclass(frozen=True)
class ImageRecord:
    path: Path
    label: str
    label_index: int
    sequence_group: str


@dataclass(frozen=True)
class DatasetLayout:
    root: Path
    train_dir: Path
    publisher_test_dir: Path
    classes: tuple[str, ...]


@dataclass(frozen=True)
class DatasetSplits:
    train: tuple[ImageRecord, ...]
    validation: tuple[ImageRecord, ...]
    test: tuple[ImageRecord, ...]


def _unwrap_single_directory(path: Path, expected_name: str) -> Path:
    nested = path / expected_name
    return nested if nested.is_dir() else path


def discover_dataset(root: str | Path = settings.DATASET_DIR) -> DatasetLayout:
    """Kaggle arşivinin tek/çift katmanlı yerleşimini güvenli biçimde keşfet."""

    root = Path(root).resolve()
    train_dir = _unwrap_single_directory(
        root / "asl_alphabet_train", "asl_alphabet_train"
    )
    publisher_test_dir = _unwrap_single_directory(
        root / "asl_alphabet_test", "asl_alphabet_test"
    )
    if not train_dir.is_dir() or not publisher_test_dir.is_dir():
        raise FileNotFoundError(
            "ASL Alphabet train/test klasörleri bulunamadı. "
            f"Kontrol edilen kök: {root}"
        )

    classes = tuple(
        label for label in settings.ASL_CLASSES if (train_dir / label).is_dir()
    )
    missing = sorted(set(settings.ASL_CLASSES) - set(classes))
    unexpected = sorted(
        path.name
        for path in train_dir.iterdir()
        if path.is_dir() and path.name not in settings.ASL_CLASSES
    )
    if missing or unexpected:
        raise ValueError(
            f"Sınıf yapısı geçersiz. Eksik={missing}, beklenmeyen={unexpected}"
        )
    return DatasetLayout(root, train_dir, publisher_test_dir, classes)


def load_label_mapping(path: str | Path = settings.LABEL_MAPPING_PATH) -> dict[str, int]:
    with Path(path).open("r", encoding="utf-8") as handle:
        document = json.load(handle)
    mapping = document["class_to_index"]
    if tuple(sorted(mapping, key=mapping.get)) != settings.ASL_CLASSES:
        raise ValueError("Etiket eşlemesi config.settings.ASL_CLASSES ile uyuşmuyor.")
    return {str(label): int(index) for label, index in mapping.items()}


def derive_sequence_group(path: Path, label: str, group_size: int) -> str:
    """Dosya sonundaki sayıyı ardışık-kare grubuna dönüştür.

    Veri setinde kişi/oturum kimliği yoktur. Bu sezgisel grup kişi bağımsızlığı
    sağlamaz; yalnız komşu karelerin farklı bölümlere saçılmasını azaltır.
    """

    if group_size <= 0:
        raise ValueError("group_size pozitif olmalıdır.")
    match = _NUMBER_PATTERN.search(path.stem)
    if not match:
        return f"{label}:{path.stem}"
    number = int(match.group(1))
    return f"{label}:{max(0, number - 1) // group_size:05d}"


def index_training_images(
    layout: DatasetLayout,
    mapping: dict[str, int] | None = None,
    *,
    group_size: int = settings.SEQUENCE_GROUP_SIZE,
) -> tuple[ImageRecord, ...]:
    """Görüntüleri RAM'e almadan yalnız dosya yollarını indeksle."""

    mapping = mapping or load_label_mapping()
    records: list[ImageRecord] = []
    for label in layout.classes:
        paths = sorted(
            (
                path
                for path in (layout.train_dir / label).iterdir()
                if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
            ),
            key=lambda path: path.name.lower(),
        )
        records.extend(
            ImageRecord(
                path=path.resolve(),
                label=label,
                label_index=mapping[label],
                sequence_group=derive_sequence_group(path, label, group_size),
            )
            for path in paths
        )
    return tuple(records)


def split_by_sequence_groups(
    records: Sequence[ImageRecord],
    *,
    train_ratio: float = settings.TRAIN_RATIO,
    validation_ratio: float = settings.VALIDATION_RATIO,
    test_ratio: float = settings.TEST_RATIO,
    seed: int = settings.RANDOM_SEED,
) -> DatasetSplits:
    """Her sınıfta ardışık dosya gruplarını bölümler arasında ayır."""

    if not np.isclose(train_ratio + validation_ratio + test_ratio, 1.0):
        raise ValueError("Bölme oranlarının toplamı 1 olmalıdır.")

    by_label_and_group: dict[str, dict[str, list[ImageRecord]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for record in records:
        by_label_and_group[record.label][record.sequence_group].append(record)

    output: dict[str, list[ImageRecord]] = {
        "train": [],
        "validation": [],
        "test": [],
    }
    rng = np.random.default_rng(seed)
    for label in sorted(by_label_and_group):
        groups = sorted(by_label_and_group[label].items())
        rng.shuffle(groups)
        group_count = len(groups)
        train_end = int(round(group_count * train_ratio))
        validation_end = train_end + int(round(group_count * validation_ratio))
        assignments = (
            ("train", groups[:train_end]),
            ("validation", groups[train_end:validation_end]),
            ("test", groups[validation_end:]),
        )
        for split_name, selected_groups in assignments:
            for _, group_records in selected_groups:
                output[split_name].extend(group_records)

    for split_name in output:
        output[split_name].sort(key=lambda record: str(record.path).lower())
    return DatasetSplits(
        train=tuple(output["train"]),
        validation=tuple(output["validation"]),
        test=tuple(output["test"]),
    )


def save_split_indexes(
    splits: DatasetSplits,
    output_dir: str | Path = settings.SPLIT_INDEX_DIR,
) -> None:
    """Görüntüleri kopyalamadan tekrar üretilebilir JSONL yol indeksleri yaz."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for split_name in ("train", "validation", "test"):
        records = getattr(splits, split_name)
        destination = output_dir / f"{split_name}.jsonl"
        with destination.open("w", encoding="utf-8", newline="\n") as handle:
            for record in records:
                payload = asdict(record)
                try:
                    payload["path"] = record.path.relative_to(
                        settings.PROJECT_ROOT
                    ).as_posix()
                    payload["path_is_project_relative"] = True
                except ValueError:
                    payload["path"] = str(record.path)
                    payload["path_is_project_relative"] = False
                handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def load_split_index(path: str | Path) -> tuple[ImageRecord, ...]:
    """JSONL bölme indeksini oku ve proje-göreli yolları çöz."""

    records: list[ImageRecord] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            image_path = Path(payload["path"])
            if payload.get("path_is_project_relative", False):
                image_path = settings.PROJECT_ROOT / image_path
            records.append(
                ImageRecord(
                    path=image_path.resolve(),
                    label=str(payload["label"]),
                    label_index=int(payload["label_index"]),
                    sequence_group=str(payload["sequence_group"]),
                )
            )
    return tuple(records)


def iter_image_batches(
    records: Sequence[ImageRecord],
    *,
    batch_size: int = 32,
    preprocess_config: PreprocessConfig | None = None,
    training: bool = False,
    shuffle: bool = False,
    seed: int = settings.RANDOM_SEED,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Tüm veri setini RAM'e almadan NumPy batch'leri üret."""

    if batch_size <= 0:
        raise ValueError("batch_size pozitif olmalıdır.")
    rng = np.random.default_rng(seed)
    indices = np.arange(len(records))
    if shuffle:
        rng.shuffle(indices)
    for start in range(0, len(indices), batch_size):
        batch_indices = indices[start : start + batch_size]
        images = [
            preprocess_image(
                records[int(index)].path,
                preprocess_config,
                training=training,
                rng=rng,
            )
            for index in batch_indices
        ]
        labels = [records[int(index)].label_index for index in batch_indices]
        yield np.stack(images), np.asarray(labels, dtype=np.int64)


def prefetch_batches(
    batches: Iterator[tuple[np.ndarray, np.ndarray]],
    *,
    buffer_size: int = 2,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Batch'leri sınırlı bir arka plan kuyruğunda önceden oku."""

    if buffer_size <= 0:
        raise ValueError("buffer_size pozitif olmalıdır.")
    output: queue.Queue[object] = queue.Queue(maxsize=buffer_size)
    sentinel = object()

    def producer() -> None:
        try:
            for batch in batches:
                output.put(batch)
        except BaseException as exc:  # Hata tüketici iş parçacığına aktarılır.
            output.put(exc)
        finally:
            output.put(sentinel)

    worker = threading.Thread(target=producer, name="image-batch-prefetch", daemon=True)
    worker.start()
    while True:
        item = output.get()
        if item is sentinel:
            break
        if isinstance(item, BaseException):
            raise item
        yield item  # type: ignore[misc]
    worker.join()


def build_tf_dataset(
    records: Sequence[ImageRecord],
    *,
    batch_size: int = 32,
    training: bool = False,
    seed: int = settings.RANDOM_SEED,
):
    """TensorFlow mevcutsa yol tabanlı, prefetch kullanan ``tf.data`` hattı kur.

    TensorFlow Aşama 2'nin hafif test ortamında zorunlu değildir; Aşama 3 temiz
    ortam kurulumunda bu fonksiyon entegrasyon testinden geçirilecektir.
    """

    try:
        import tensorflow as tf
    except ImportError as exc:  # pragma: no cover - kurulum bağımlı dal
        raise RuntimeError(
            "TensorFlow kurulu değil. Tam requirements.txt ortamını kurun."
        ) from exc

    paths = [str(record.path) for record in records]
    labels = [record.label_index for record in records]
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        dataset = dataset.shuffle(
            min(len(records), 10_000), seed=seed, reshuffle_each_iteration=True
        )

    target_height, target_width = settings.IMAGE_SIZE

    dataset = dataset.enumerate()

    def decode(index, payload):
        path, label = payload
        contents = tf.io.read_file(path)
        image = tf.io.decode_jpeg(contents, channels=3)
        image = tf.image.resize_with_pad(image, target_height, target_width)
        image = tf.cast(image, tf.float32) / 255.0
        if training:
            random_seed = tf.stack(
                [tf.cast(seed, tf.int64), tf.cast(index, tf.int64)]
            )
            image = tf.image.stateless_random_brightness(
                image, max_delta=0.15, seed=random_seed
            )
            image = tf.image.stateless_random_contrast(
                image,
                lower=0.85,
                upper=1.15,
                seed=random_seed + tf.constant([0, 1], dtype=tf.int64),
            )
            image = tf.clip_by_value(image, 0.0, 1.0)
        return image, label

    dataset = dataset.map(decode, num_parallel_calls=tf.data.AUTOTUNE)
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

