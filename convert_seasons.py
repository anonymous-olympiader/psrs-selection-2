# Photo1.jpg -> Summer.jpg, Photo2.jpg -> Autumn.jpg

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np


def _build_hue_map_autumn_to_summer() -> np.ndarray:
    # OpenCV Hue в [0, 180]
    # Осень: красный 0-10, 170-180; оранжевый 10-25; жёлтый 25-50
    # Лето: зелёный примерно 35-85
    lut = np.arange(256, dtype=np.uint8)
    out = np.zeros(256, dtype=np.uint8)
    for i in range(256):
        h = min(i, 180)
        if h <= 50 or h >= 150:  # красный, оранжевый, жёлтый
            # Линейно отображаем на зелёный диапазон [40, 80]
            if h <= 50:
                t = h / 50.0
                out[i] = int(40 + t * 40)  # 40..80
            else:
                t = (180 - h) / 30.0
                out[i] = int(40 + t * 40)
            out[i] = min(out[i], 180)
        else:
            out[i] = i  # зелёные и так оставляем
    return out


def _build_hue_map_summer_to_autumn() -> np.ndarray:
    # Зелёный 35-85 -> оранжево-красный 8-28
    out = np.zeros(256, dtype=np.uint8)
    for i in range(256):
        h = min(i, 180)
        if 30 <= h <= 95:  # зелёный диапазон
            t = (h - 30) / 65.0
            out[i] = int(8 + t * 20)  # 8..28 (оранжевый-красный)
        else:
            out[i] = i
    return out


def _foliage_mask(hsv: np.ndarray, hue_lo: int, hue_hi: int,
                  sat_min: int = 25, val_min: int = 30) -> np.ndarray:
    h, s, v = cv2.split(hsv)
    if hue_lo <= hue_hi:
        hue_ok = (h >= hue_lo) & (h <= hue_hi)
    else:
        hue_ok = (h >= hue_lo) | (h <= hue_hi)
    sat_ok = s >= sat_min
    val_ok = v >= val_min
    mask = (hue_ok & sat_ok & val_ok).astype(np.uint8)
    mask = cv2.GaussianBlur(mask, (5, 5), 1.0)
    return mask


def apply_season_transform(hsv: np.ndarray, to_summer: bool) -> np.ndarray:
    h, s, v = cv2.split(hsv)
    h = h.astype(np.uint8)

    if to_summer:
        hue_lut = _build_hue_map_autumn_to_summer()
        foliage = _foliage_mask(hsv, 0, 50, 25, 30) | _foliage_mask(hsv, 150, 180, 25, 30)
    else:
        hue_lut = _build_hue_map_summer_to_autumn()
        foliage = _foliage_mask(hsv, 30, 95, 25, 30)

    h_new = cv2.LUT(h, hue_lut)
    blend = foliage.astype(np.float32) / 255.0
    h_out = (h_new.astype(np.float32) * blend + h.astype(np.float32) * (1 - blend)).astype(np.uint8)
    return cv2.merge([h_out, s, v])


def autumn_to_summer(bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    hsv_out = apply_season_transform(hsv, to_summer=True)
    return cv2.cvtColor(hsv_out, cv2.COLOR_HSV2BGR)


def summer_to_autumn(bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    hsv_out = apply_season_transform(hsv, to_summer=False)
    return cv2.cvtColor(hsv_out, cv2.COLOR_HSV2BGR)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input_dir",
        nargs="?",
        default=".",
        help="Каталог с Photo1.jpg и Photo2.jpg (по умолчанию текущий)",
    )
    parser.add_argument(
        "-o", "--output-dir",
        default=None,
        help="Каталог для Summer.jpg и Autumn.jpg (по умолчанию совпадает с input_dir)",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir else input_dir
    photo1 = input_dir / "Photo1.jpg"
    photo2 = input_dir / "Photo2.jpg"

    if not photo1.is_file():
        print(f"Ошибка: не найден файл {photo1}", file=sys.stderr)
        return 1
    if not photo2.is_file():
        print(f"Ошибка: не найден файл {photo2}", file=sys.stderr)
        return 1

    img_autumn = cv2.imread(str(photo1))
    img_summer_src = cv2.imread(str(photo2))
    if img_autumn is None:
        print(f"Ошибка: не удалось загрузить {photo1}", file=sys.stderr)
        return 1
    if img_summer_src is None:
        print(f"Ошибка: не удалось загрузить {photo2}", file=sys.stderr)
        return 1

    # Photo1 (осень) -> Summer.jpg (лето)
    summer_img = autumn_to_summer(img_autumn)
    summer_path = output_dir / "Summer.jpg"
    cv2.imwrite(str(summer_path), summer_img)
    print(f"Сохранено: {summer_path} (из Photo1.jpg)")

    # Photo2 (лето) -> Autumn.jpg (осень)
    autumn_img = summer_to_autumn(img_summer_src)
    autumn_path = output_dir / "Autumn.jpg"
    cv2.imwrite(str(autumn_path), autumn_img)
    print(f"Сохранено: {autumn_path} (из Photo2.jpg)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
