"""Gerçek ASL Alphabet dosyalarını tarayıp tekrarlanabilir analiz üretir."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError

from config import settings
from recognition.dataset_loader import discover_dataset


@dataclass(frozen=True)
class FileInspection:
    path: str
    label: str
    subset: str
    byte_count: int
    image_format: str | None
    mode: str | None
    width: int | None
    height: int | None
    sha256: str | None
    difference_hash: int | None
    error: str | None


def _difference_hash(image: Image.Image) -> int:
    gray = image.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
    pixels = np.asarray(gray, dtype=np.int16)
    bits = pixels[:, 1:] > pixels[:, :-1]
    value = 0
    for bit in bits.flat:
        value = (value << 1) | int(bit)
    return value


def _inspect(arguments: tuple[Path, str, str, Path]) -> FileInspection:
    path, label, subset, base = arguments
    try:
        payload = path.read_bytes()
        with Image.open(BytesIO(payload)) as image:
            image.load()
            image_format = image.format
            mode = image.mode
            width, height = image.size
            difference_hash = _difference_hash(image)
        return FileInspection(
            path=str(path.relative_to(base)),
            label=label,
            subset=subset,
            byte_count=len(payload),
            image_format=image_format,
            mode=mode,
            width=width,
            height=height,
            sha256=hashlib.sha256(payload).hexdigest(),
            difference_hash=difference_hash,
            error=None,
        )
    except (OSError, UnidentifiedImageError, ValueError) as exc:
        return FileInspection(
            path=str(path.relative_to(base)),
            label=label,
            subset=subset,
            byte_count=path.stat().st_size if path.exists() else 0,
            image_format=None,
            mode=None,
            width=None,
            height=None,
            sha256=None,
            difference_hash=None,
            error=f"{type(exc).__name__}: {exc}",
        )


def _numeric_suffix(path_text: str) -> int | None:
    stem = Path(path_text).stem
    digits = ""
    for character in reversed(stem):
        if not character.isdigit():
            break
        digits = character + digits
    return int(digits) if digits else None


def _hamming_distance(left: int, right: int) -> int:
    return (left ^ right).bit_count()


def analyze_dataset(
    root: str | Path = settings.DATASET_DIR,
    *,
    workers: int = 8,
) -> dict[str, object]:
    """Tüm gerçek görüntüleri aç, hash'le ve özet istatistik döndür."""

    layout = discover_dataset(root)
    tasks: list[tuple[Path, str, str, Path]] = []
    class_counts: Counter[str] = Counter()
    for label in layout.classes:
        for path in sorted((layout.train_dir / label).glob("*")):
            if path.is_file():
                tasks.append((path, label, "train_source", layout.root))
                class_counts[label] += 1

    publisher_test_labels: list[str] = []
    for path in sorted(layout.publisher_test_dir.glob("*")):
        if path.is_file():
            label = path.stem.removesuffix("_test")
            publisher_test_labels.append(label)
            tasks.append((path, label, "publisher_test", layout.root))

    inspections: list[FileInspection] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for index, inspection in enumerate(executor.map(_inspect, tasks), start=1):
            inspections.append(inspection)
            if index % 10_000 == 0:
                print(f"Taranan dosya: {index}/{len(tasks)}", flush=True)

    valid = [inspection for inspection in inspections if inspection.error is None]
    corrupt = [inspection for inspection in inspections if inspection.error is not None]
    sha_groups: dict[str, list[str]] = defaultdict(list)
    dhash_groups: dict[tuple[str, int], list[str]] = defaultdict(list)
    for inspection in valid:
        assert inspection.sha256 is not None
        assert inspection.difference_hash is not None
        sha_groups[inspection.sha256].append(inspection.path)
        dhash_groups[(inspection.label, inspection.difference_hash)].append(
            inspection.path
        )

    exact_duplicate_groups = [paths for paths in sha_groups.values() if len(paths) > 1]
    same_dhash_groups = [paths for paths in dhash_groups.values() if len(paths) > 1]
    subset_by_path = {inspection.path: inspection.subset for inspection in inspections}
    cross_subset_exact_groups = [
        paths
        for paths in exact_duplicate_groups
        if len({subset_by_path[path] for path in paths}) > 1
    ]
    duplicated_publisher_test_paths = {
        path
        for paths in cross_subset_exact_groups
        for path in paths
        if subset_by_path[path] == "publisher_test"
    }

    adjacent_distances: list[int] = []
    by_label: dict[str, list[FileInspection]] = defaultdict(list)
    for inspection in valid:
        if inspection.subset == "train_source":
            by_label[inspection.label].append(inspection)
    for label_inspections in by_label.values():
        numbered = [
            (number, inspection)
            for inspection in label_inspections
            if (number := _numeric_suffix(inspection.path)) is not None
        ]
        numbered.sort(key=lambda pair: pair[0])
        for (left_number, left), (right_number, right) in zip(
            numbered, numbered[1:]
        ):
            if right_number == left_number + 1:
                assert left.difference_hash is not None
                assert right.difference_hash is not None
                adjacent_distances.append(
                    _hamming_distance(left.difference_hash, right.difference_hash)
                )

    test_missing = sorted(set(settings.ASL_CLASSES) - set(publisher_test_labels))
    test_unexpected = sorted(set(publisher_test_labels) - set(settings.ASL_CLASSES))
    dimensions = Counter((item.width, item.height) for item in valid)
    formats = Counter(item.image_format for item in valid)
    modes = Counter(item.mode for item in valid)

    return {
        "dataset_root": str(layout.root),
        "total_files": len(inspections),
        "valid_images": len(valid),
        "corrupt_images": len(corrupt),
        "corrupt_file_details": [asdict(item) for item in corrupt],
        "train_source_images": sum(class_counts.values()),
        "publisher_test_images": len(publisher_test_labels),
        "class_count": len(layout.classes),
        "classes": list(layout.classes),
        "class_counts": dict(class_counts),
        "publisher_test_labels": publisher_test_labels,
        "publisher_test_missing_classes": test_missing,
        "publisher_test_unexpected_classes": test_unexpected,
        "formats": {str(key): value for key, value in formats.items()},
        "color_modes": {str(key): value for key, value in modes.items()},
        "dimensions": {f"{key[0]}x{key[1]}": value for key, value in dimensions.items()},
        "total_image_bytes": sum(item.byte_count for item in inspections),
        "exact_duplicate_group_count": len(exact_duplicate_groups),
        "exact_duplicate_excess_file_count": sum(
            len(group) - 1 for group in exact_duplicate_groups
        ),
        "cross_subset_exact_duplicate_group_count": len(cross_subset_exact_groups),
        "publisher_test_exact_duplicate_count": len(
            duplicated_publisher_test_paths
        ),
        "exact_duplicate_examples": exact_duplicate_groups[:20],
        "same_dhash_candidate_group_count": len(same_dhash_groups),
        "same_dhash_candidate_excess_file_count": sum(
            len(group) - 1 for group in same_dhash_groups
        ),
        "same_dhash_candidate_examples": same_dhash_groups[:20],
        "adjacent_numeric_pair_count": len(adjacent_distances),
        "adjacent_dhash_distance_median": (
            statistics.median(adjacent_distances) if adjacent_distances else None
        ),
        "adjacent_pairs_hamming_le_2": sum(
            distance <= 2 for distance in adjacent_distances
        ),
        "adjacent_pairs_hamming_le_5": sum(
            distance <= 5 for distance in adjacent_distances
        ),
        "similarity_method_note": (
            "dHash aynı-değer grupları ve yalnız sayısal olarak ardışık dosya "
            "çiftlerinin Hamming uzaklığı tarandı; bu, tüm O(n²) yakın-komşu "
            "aramasının yerine geçen aday taramasıdır."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=settings.DATASET_DIR)
    parser.add_argument(
        "--output",
        type=Path,
        default=settings.PROCESSED_DATA_DIR / "dataset_analysis.json",
    )
    parser.add_argument("--workers", type=int, default=8)
    arguments = parser.parse_args()
    report = analyze_dataset(arguments.root, workers=arguments.workers)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

