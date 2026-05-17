from __future__ import annotations

import argparse
import os
import random
from pathlib import Path

import cv2
import numpy as np
from onnxruntime.quantization import (
    CalibrationDataReader,
    QuantFormat,
    QuantType,
    quantize_static,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_FP32 = ROOT / "helmet_v2" / "weights" / "best.onnx"
DEFAULT_INT8 = ROOT / "helmetv2_int8.onnx"
DEFAULT_CALIB_DIR = ROOT / "helmet_dataset_rf" / "v2" / "train" / "images"
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")


def resolve_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else ROOT / path


def normalize_imgsz(values: list[int]) -> tuple[int, int]:
    if len(values) == 1:
        return values[0], values[0]
    if len(values) == 2:
        return values[0], values[1]
    raise argparse.ArgumentTypeError("--imgsz expects one value or two values: height width")


def letterbox(im: np.ndarray, new_shape: tuple[int, int], color: tuple[int, int, int] = (114, 114, 114)) -> np.ndarray:
    shape = im.shape[:2]
    new_h, new_w = new_shape
    ratio = min(new_h / shape[0], new_w / shape[1])

    new_unpad = (int(round(shape[1] * ratio)), int(round(shape[0] * ratio)))
    dw = (new_w - new_unpad[0]) / 2
    dh = (new_h - new_unpad[1]) / 2

    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)

    top = int(round(dh - 0.1))
    bottom = int(round(dh + 0.1))
    left = int(round(dw - 0.1))
    right = int(round(dw + 0.1))

    return cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)


class YOLOCalibrationDataReader(CalibrationDataReader):
    def __init__(self, image_dirs: list[Path], input_name: str, imgsz: tuple[int, int], max_images: int = 300):
        self.input_name = input_name
        self.imgsz = imgsz
        self.image_paths: list[Path] = []
        self.index = 0

        for image_dir in image_dirs:
            if not image_dir.is_dir():
                continue
            for root, _, files in os.walk(image_dir):
                for filename in files:
                    if filename.lower().endswith(IMAGE_EXTENSIONS):
                        self.image_paths.append(Path(root) / filename)

        if not self.image_paths:
            raise RuntimeError(f"No calibration images found in: {image_dirs}")

        random.Random(42).shuffle(self.image_paths)
        self.image_paths = self.image_paths[:max_images]

        print(f"Calibration dirs: {[str(p) for p in image_dirs]}")
        print(f"Calibration images: {len(self.image_paths)}")

    def _preprocess_image(self, image_path: Path) -> np.ndarray | None:
        data = np.fromfile(str(image_path), dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img is None:
            print(f"Warning: cannot read image: {image_path}")
            return None

        img = letterbox(img, self.imgsz)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        return np.expand_dims(img, axis=0)

    def get_next(self):
        while self.index < len(self.image_paths):
            image_path = self.image_paths[self.index]
            self.index += 1
            arr = self._preprocess_image(image_path)
            if arr is not None:
                return {self.input_name: arr}
        return None


def quantize_one(
    fp32_path: Path,
    int8_path: Path,
    calib_dirs: list[Path],
    imgsz: tuple[int, int],
    input_name: str,
    max_images: int,
) -> None:
    print("\n" + "=" * 60)
    print(f"FP32 ONNX: {fp32_path}")
    print(f"INT8 ONNX: {int8_path}")
    print(f"Input size: {imgsz}")
    print("=" * 60)

    if not fp32_path.is_file():
        raise FileNotFoundError(f"Cannot find ONNX model: {fp32_path}")

    int8_path.parent.mkdir(parents=True, exist_ok=True)
    data_reader = YOLOCalibrationDataReader(
        image_dirs=calib_dirs,
        input_name=input_name,
        imgsz=imgsz,
        max_images=max_images,
    )

    quantize_static(
        model_input=str(fp32_path),
        model_output=str(int8_path),
        calibration_data_reader=data_reader,
        quant_format=QuantFormat.QOperator,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QUInt8,
        per_channel=True,
        reduce_range=False,
        op_types_to_quantize=["Conv", "MatMul"],
    )

    print(f"Quantization complete: {int8_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Static INT8 quantization for a YOLOv5 ONNX model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--fp32", default=str(DEFAULT_FP32), help="Input FP32 ONNX model.")
    parser.add_argument("--int8", default=str(DEFAULT_INT8), help="Output INT8 ONNX model.")
    parser.add_argument(
        "--calib-dir",
        nargs="+",
        default=[str(DEFAULT_CALIB_DIR)],
        help="Calibration image directory or directories.",
    )
    parser.add_argument("--imgsz", nargs="+", type=int, default=[320, 320], help="Model input size: height width.")
    parser.add_argument("--input-name", default="images", help="ONNX input tensor name.")
    parser.add_argument("--max-images", type=int, default=1000, help="Maximum calibration images to use.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    quantize_one(
        fp32_path=resolve_path(args.fp32),
        int8_path=resolve_path(args.int8),
        calib_dirs=[resolve_path(path) for path in args.calib_dir],
        imgsz=normalize_imgsz(args.imgsz),
        input_name=args.input_name,
        max_images=args.max_images,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
