from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEFAULT_WEIGHTS = ROOT / "helmet_v2" / "weights" / "best.pt"
DEFAULT_DATA = ROOT / "data.yaml"


def resolve_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else ROOT / path


def normalize_imgsz(values: list[int]) -> tuple[int, int]:
    if len(values) == 1:
        return values[0], values[0]
    if len(values) == 2:
        return values[0], values[1]
    raise argparse.ArgumentTypeError("--imgsz expects one value or two values: height width")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export a trained YOLOv5 .pt model to ONNX.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--weights",
        default=str(DEFAULT_WEIGHTS),
        help="Path to the YOLOv5 .pt weights file.",
    )
    parser.add_argument(
        "--data",
        default=str(DEFAULT_DATA),
        help="Path to dataset yaml. It is passed to yolov5/export.py.",
    )
    parser.add_argument(
        "--imgsz",
        "--img",
        nargs="+",
        type=int,
        default=[320, 320],
        help="ONNX input image size. Use 320 320 to match the current INT8 script.",
    )
    parser.add_argument("--batch-size", type=int, default=1, help="Export batch size.")
    parser.add_argument("--device", default="cpu", help="Export device, for example cpu or 0.")
    parser.add_argument("--opset", type=int, default=12, help="ONNX opset version.")
    parser.add_argument("--simplify", action="store_true", help="Run ONNX simplifier during export.")
    parser.add_argument("--dynamic", action="store_true", help="Export ONNX with dynamic axes.")
    parser.add_argument(
        "--output",
        default="",
        help="Optional output .onnx path. By default YOLOv5 writes next to the weights file.",
    )
    parser.add_argument(
        "--yolov5-dir",
        default=str(ROOT / "yolov5"),
        help="Path to the local YOLOv5 directory that contains export.py.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()

    yolov5_dir = resolve_path(args.yolov5_dir)
    export_py = yolov5_dir / "export.py"
    weights = resolve_path(args.weights)
    data = resolve_path(args.data)
    imgsz_h, imgsz_w = normalize_imgsz(args.imgsz)

    if not export_py.is_file():
        raise FileNotFoundError(f"Cannot find YOLOv5 export.py: {export_py}")
    if not weights.is_file():
        raise FileNotFoundError(f"Cannot find weights file: {weights}")
    if not data.is_file():
        raise FileNotFoundError(f"Cannot find dataset yaml: {data}")

    command = [
        sys.executable,
        str(export_py),
        "--weights",
        str(weights),
        "--data",
        str(data),
        "--imgsz",
        str(imgsz_h),
        str(imgsz_w),
        "--batch-size",
        str(args.batch_size),
        "--device",
        args.device,
        "--include",
        "onnx",
        "--opset",
        str(args.opset),
    ]

    if args.simplify:
        command.append("--simplify")
    if args.dynamic:
        command.append("--dynamic")

    print("Running export command:")
    print(" ".join(f'"{part}"' if " " in part else part for part in command))
    subprocess.run(command, cwd=str(yolov5_dir), check=True)

    produced = weights.with_suffix(".onnx")
    if args.output:
        output = resolve_path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        if produced.resolve() != output.resolve():
            shutil.copy2(produced, output)
        print(f"ONNX exported to: {output}")
    else:
        print(f"ONNX exported to: {produced}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
