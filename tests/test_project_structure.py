"""Aşama 1 proje sözleşmelerinin bağımsız yapısal testleri."""

import unittest
from pathlib import Path

from config import settings


class ProjectStructureTests(unittest.TestCase):
    def test_required_directories_exist(self) -> None:
        required = (
            "assets",
            "config",
            "data/raw",
            "data/processed",
            "docs",
            "models",
            "nlp",
            "recognition",
            "tests",
            "text_builder",
        )
        for relative_path in required:
            with self.subTest(path=relative_path):
                self.assertTrue(
                    (settings.PROJECT_ROOT / relative_path).is_dir(),
                    f"Eksik dizin: {relative_path}",
                )

    def test_asl_class_contract(self) -> None:
        self.assertEqual(len(settings.ASL_LETTER_CLASSES), 26)
        self.assertEqual(len(settings.ASL_CLASSES), settings.EXPECTED_CLASS_COUNT)
        self.assertEqual(set(settings.CONTROL_CLASSES), {"del", "nothing", "space"})
        self.assertEqual(len(set(settings.ASL_CLASSES)), settings.EXPECTED_CLASS_COUNT)

    def test_paths_are_inside_project(self) -> None:
        project_root = settings.PROJECT_ROOT.resolve()
        for path in (
            settings.DATA_DIR,
            settings.RAW_DATA_DIR,
            settings.PROCESSED_DATA_DIR,
            settings.MODEL_DIR,
            settings.ASSET_DIR,
        ):
            with self.subTest(path=path):
                self.assertTrue(path.resolve().is_relative_to(project_root))


if __name__ == "__main__":
    unittest.main()

