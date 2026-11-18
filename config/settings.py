"""Proje genelinde paylaşılacak sabitler ve dosya yolları."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODEL_DIR = PROJECT_ROOT / "models"
ASSET_DIR = PROJECT_ROOT / "assets"

SOURCE_IMAGE_SIZE = (200, 200)
# 224x224, MobileNet/EfficientNet gibi CPU dostu transfer öğrenme adaylarıyla
# karşılaştırma yapmayı kolaylaştırır. Yeniden boyutlandırma en-boy oranını korur.
IMAGE_SIZE = (224, 224)
ASL_LETTER_CLASSES = tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
CONTROL_CLASSES = ("del", "nothing", "space")
ASL_CLASSES = ASL_LETTER_CLASSES + CONTROL_CLASSES
EXPECTED_CLASS_COUNT = 29

RANDOM_SEED = 42
TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15
TEST_RATIO = 0.15
# Dosya adlarındaki ardışık kareleri aynı grupta tutmak için sezgisel blok.
SEQUENCE_GROUP_SIZE = 50

DATASET_DIR = RAW_DATA_DIR / "asl-alphabet"
TRAIN_DATA_DIR = DATASET_DIR / "asl_alphabet_train" / "asl_alphabet_train"
PUBLISHER_TEST_DIR = DATASET_DIR / "asl_alphabet_test" / "asl_alphabet_test"
LABEL_MAPPING_PATH = PROJECT_ROOT / "config" / "label_mapping.json"
SPLIT_INDEX_DIR = PROCESSED_DATA_DIR / "splits"

DATASET_SLUG = "grassknoted/asl-alphabet"
DATASET_URL = "https://www.kaggle.com/datasets/grassknoted/asl-alphabet"

