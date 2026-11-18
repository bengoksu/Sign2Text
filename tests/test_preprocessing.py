"""Görüntü ön işleme birim testleri."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from recognition.preprocessing import (
    ImageReadError,
    PreprocessConfig,
    augment_image,
    preprocess_image,
    safe_read_image,
)


class PreprocessingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_rgb_output_shape_and_normalized_range(self) -> None:
        path = self.root / "grayscale.png"
        Image.new("L", (80, 40), color=128).save(path)
        output = preprocess_image(path, PreprocessConfig(image_size=(64, 96)))
        self.assertEqual(output.shape, (64, 96, 3))
        self.assertEqual(output.dtype, np.float32)
        self.assertGreaterEqual(float(output.min()), 0.0)
        self.assertLessEqual(float(output.max()), 1.0)

    def test_rgba_is_read_as_rgb(self) -> None:
        path = self.root / "rgba.png"
        Image.new("RGBA", (20, 30), color=(10, 20, 30, 100)).save(path)
        image = safe_read_image(path)
        self.assertEqual(image.mode, "RGB")

    def test_corrupt_image_raises_domain_error(self) -> None:
        path = self.root / "broken.jpg"
        path.write_bytes(b"not an image")
        with self.assertRaises(ImageReadError):
            preprocess_image(path)

    def test_augmentation_is_seed_reproducible_and_keeps_size(self) -> None:
        image = Image.new("RGB", (64, 64), color=(120, 80, 40))
        first = np.asarray(augment_image(image, np.random.default_rng(7)))
        second = np.asarray(augment_image(image, np.random.default_rng(7)))
        self.assertEqual(first.shape, (64, 64, 3))
        np.testing.assert_array_equal(first, second)


if __name__ == "__main__":
    unittest.main()

