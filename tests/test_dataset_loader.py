"""Yol tabanlı veri yükleme ve bölme testleri."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from config import settings
from recognition.dataset_loader import (
    ImageRecord,
    discover_dataset,
    iter_image_batches,
    load_label_mapping,
    load_split_index,
    prefetch_batches,
    save_split_indexes,
    split_by_sequence_groups,
)
from recognition.preprocessing import PreprocessConfig


class DatasetLoaderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def _records(self) -> tuple[ImageRecord, ...]:
        records = []
        for label_index, label in enumerate(("A", "B")):
            for group in range(4):
                for item in range(2):
                    path = self.root / f"{label}_{group}_{item}.jpg"
                    Image.new(
                        "RGB", (20 + item, 30), color=(group * 20, 50, 100)
                    ).save(path)
                    records.append(
                        ImageRecord(path, label, label_index, f"{label}:{group}")
                    )
        return tuple(records)

    def test_label_mapping_contract(self) -> None:
        mapping = load_label_mapping()
        self.assertEqual(len(mapping), 29)
        self.assertEqual(mapping["A"], 0)
        self.assertEqual(mapping["Z"], 25)
        self.assertEqual(mapping["del"], 26)
        self.assertEqual(mapping["nothing"], 27)
        self.assertEqual(mapping["space"], 28)

    def test_group_split_has_no_path_or_group_overlap(self) -> None:
        splits = split_by_sequence_groups(
            self._records(),
            train_ratio=0.5,
            validation_ratio=0.25,
            test_ratio=0.25,
            seed=3,
        )
        path_sets = [
            {record.path for record in split}
            for split in (splits.train, splits.validation, splits.test)
        ]
        group_sets = [
            {record.sequence_group for record in split}
            for split in (splits.train, splits.validation, splits.test)
        ]
        for left in range(3):
            for right in range(left + 1, 3):
                self.assertTrue(path_sets[left].isdisjoint(path_sets[right]))
                self.assertTrue(group_sets[left].isdisjoint(group_sets[right]))

    def test_batch_loader_reads_on_demand(self) -> None:
        records = self._records()[:5]
        batches = iter_image_batches(
            records,
            batch_size=4,
            preprocess_config=PreprocessConfig(image_size=(32, 32)),
            shuffle=False,
        )
        images, labels = next(batches)
        self.assertEqual(images.shape, (4, 32, 32, 3))
        self.assertEqual(labels.shape, (4,))
        self.assertTrue(np.all((0.0 <= images) & (images <= 1.0)))
        final_images, _ = next(batches)
        self.assertEqual(final_images.shape[0], 1)

    def test_prefetch_batch_loader(self) -> None:
        batches = iter_image_batches(
            self._records()[:5],
            batch_size=2,
            preprocess_config=PreprocessConfig(image_size=(24, 24)),
        )
        prefetched = list(prefetch_batches(batches, buffer_size=2))
        self.assertEqual([images.shape[0] for images, _ in prefetched], [2, 2, 1])

    def test_split_index_round_trip(self) -> None:
        splits = split_by_sequence_groups(
            self._records(),
            train_ratio=0.5,
            validation_ratio=0.25,
            test_ratio=0.25,
            seed=3,
        )
        output = self.root / "indexes"
        save_split_indexes(splits, output)
        loaded = load_split_index(output / "train.jsonl")
        self.assertEqual(loaded, splits.train)

    def test_real_dataset_layout_when_available(self) -> None:
        if not settings.DATASET_DIR.exists():
            self.skipTest("Gerçek veri seti bu ortamda yok.")
        layout = discover_dataset()
        self.assertEqual(layout.classes, settings.ASL_CLASSES)
        self.assertEqual(len(list(layout.publisher_test_dir.glob("*.jpg"))), 28)


if __name__ == "__main__":
    unittest.main()

