"""CNN eğitimi ve canlı tahmin için ortak görüntü ön işleme işlemleri."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias

import numpy as np
from PIL import Image, ImageEnhance, ImageOps, UnidentifiedImageError

from config.settings import IMAGE_SIZE

ImageSource: TypeAlias = str | Path | Image.Image


class ImageReadError(ValueError):
    """Bir görüntü güvenli biçimde okunamadığında üretilir."""


@dataclass(frozen=True)
class PreprocessConfig:
    """Model girdisi için deterministik dönüşüm ayarları."""

    image_size: tuple[int, int] = IMAGE_SIZE
    normalize: bool = True
    pad_color: tuple[int, int, int] = (0, 0, 0)


@dataclass(frozen=True)
class AugmentationConfig:
    """İşaret anlamını koruyacak hafif eğitim dönüşümleri.

    Yatay çevirme bilinçli olarak desteklenmez; el yönünü ve işaretin anlamını
    değiştirebilir. Artırma yalnız eğitim batch'lerinde çağrılmalıdır.
    """

    brightness_range: tuple[float, float] = (0.85, 1.15)
    contrast_range: tuple[float, float] = (0.85, 1.15)
    rotation_degrees: float = 10.0
    scale_range: tuple[float, float] = (0.90, 1.10)
    translation_fraction: float = 0.05


def safe_read_image(source: ImageSource) -> Image.Image:
    """Girdiyi EXIF yönü uygulanmış bağımsız bir RGB görüntü olarak döndür."""

    if isinstance(source, Image.Image):
        return ImageOps.exif_transpose(source).convert("RGB").copy()

    path = Path(source)
    try:
        with Image.open(path) as image:
            image.load()
            return ImageOps.exif_transpose(image).convert("RGB").copy()
    except (FileNotFoundError, OSError, UnidentifiedImageError) as exc:
        raise ImageReadError(f"Görüntü okunamadı: {path}") from exc


def resize_with_padding(
    image: Image.Image,
    output_size: tuple[int, int],
    pad_color: tuple[int, int, int] = (0, 0, 0),
) -> Image.Image:
    """En-boy oranını koruyarak görüntüyü hedef boyuta padding ile yerleştir."""

    target_height, target_width = output_size
    if target_height <= 0 or target_width <= 0:
        raise ValueError("Çıktı boyutları pozitif olmalıdır.")

    contained = ImageOps.contain(
        image.convert("RGB"),
        (target_width, target_height),
        method=Image.Resampling.LANCZOS,
    )
    canvas = Image.new("RGB", (target_width, target_height), pad_color)
    offset = (
        (target_width - contained.width) // 2,
        (target_height - contained.height) // 2,
    )
    canvas.paste(contained, offset)
    return canvas


def _scale_and_center(image: Image.Image, scale: float) -> Image.Image:
    width, height = image.size
    scaled_size = (
        max(1, int(round(width * scale))),
        max(1, int(round(height * scale))),
    )
    scaled = image.resize(scaled_size, Image.Resampling.BILINEAR)
    if scale >= 1.0:
        left = (scaled.width - width) // 2
        top = (scaled.height - height) // 2
        return scaled.crop((left, top, left + width, top + height))

    canvas = Image.new("RGB", (width, height), (0, 0, 0))
    canvas.paste(scaled, ((width - scaled.width) // 2, (height - scaled.height) // 2))
    return canvas


def augment_image(
    image: Image.Image,
    rng: np.random.Generator,
    config: AugmentationConfig | None = None,
) -> Image.Image:
    """Bir RGB görüntüye tohumlanabilir, hafif eğitim artırması uygula."""

    config = config or AugmentationConfig()
    result = image.convert("RGB")
    result = ImageEnhance.Brightness(result).enhance(
        float(rng.uniform(*config.brightness_range))
    )
    result = ImageEnhance.Contrast(result).enhance(
        float(rng.uniform(*config.contrast_range))
    )
    result = result.rotate(
        float(rng.uniform(-config.rotation_degrees, config.rotation_degrees)),
        resample=Image.Resampling.BILINEAR,
        fillcolor=(0, 0, 0),
    )
    result = _scale_and_center(result, float(rng.uniform(*config.scale_range)))

    max_dx = result.width * config.translation_fraction
    max_dy = result.height * config.translation_fraction
    dx = float(rng.uniform(-max_dx, max_dx))
    dy = float(rng.uniform(-max_dy, max_dy))
    return result.transform(
        result.size,
        Image.Transform.AFFINE,
        (1.0, 0.0, -dx, 0.0, 1.0, -dy),
        resample=Image.Resampling.BILINEAR,
        fillcolor=(0, 0, 0),
    )


def preprocess_image(
    source: ImageSource,
    config: PreprocessConfig | None = None,
    *,
    training: bool = False,
    rng: np.random.Generator | None = None,
    augmentation: AugmentationConfig | None = None,
) -> np.ndarray:
    """Görüntüyü ``(height, width, 3)`` model tensoruna dönüştür.

    Normalizasyon açıkken çıktı float32 ve [0, 1] aralığındadır. Transfer öğrenme
    modeline özgü normalizasyon Aşama 3'te model katmanı olarak ayrıca seçilebilir.
    """

    config = config or PreprocessConfig()
    image = safe_read_image(source)
    image = resize_with_padding(image, config.image_size, config.pad_color)
    if training:
        image = augment_image(image, rng or np.random.default_rng(), augmentation)

    array = np.asarray(image, dtype=np.float32 if config.normalize else np.uint8)
    if config.normalize:
        array /= np.float32(255.0)
    return array

