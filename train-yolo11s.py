#!/usr/bin/env python3
"""
YOLO11s training pipeline
Roboflow download → local training → ONNX export for Isaac ROS / trtexec

After training, copy the .onnx to the Jetson and run following command inside of isaac-ros:
        trtexec --onnx=${ISAAC_ROS_WS}/isaac_ros_assets/models/yolo11/yolo11s.onnx \
				--saveEngine=${ISAAC_ROS_WS}/isaac_ros_assets/models/yolo11/yolo11s_fp16.plan \
				--fp16
Then use best.plan with isaac_ros_yolov8.
"""

from getpass import getpass
from pathlib import Path
from roboflow import Roboflow
from ultralytics import YOLO


# ── CONFIG ────────────────────────────────────────────────────────────────────
ROBOFLOW_WORKSPACE = "rhitcv"       # shown in your Roboflow URL
ROBOFLOW_PROJECT   = "Icon Detection Test"         # project slug, not display name
ROBOFLOW_VERSION   = 4                      # integer version number to export

PROJECT_NAME = "yolo11s_realsense"          # groups runs under runs/<PROJECT_NAME>/
RUN_NAME     = "v1"                         # runs/<PROJECT_NAME>/<RUN_NAME>/
# ─────────────────────────────────────────────────────────────────────────────


def download_dataset() -> Path:
    api_key = getpass("Roboflow API key: ")
    rf = Roboflow(api_key=api_key)
    dataset = (
        rf.workspace(ROBOFLOW_WORKSPACE)
          .project(ROBOFLOW_PROJECT)
          .version(ROBOFLOW_VERSION)
          .download("yolov11")
    )
    return Path(dataset.location) / "data.yaml"


def train(data_yaml: Path) -> Path:
    model = YOLO("yolo11s.pt")  # downloads pretrained weights on first run

    results = model.train(
        data=str(data_yaml),
        epochs=150,
        patience=50,        # early stop if val mAP doesn't improve for 50 epochs
        imgsz=640,
        batch=16,           # drop to 8 if you run out of VRAM
        device=0,           # 0 = first GPU; use "cpu" if no CUDA
        rect=True,          # efficient batching for uniform 640×480 inputs —
                            # assumes Isaac ROS letterboxes (preserves aspect ratio)
                            # rather than stretches to 640×640. If you confirm it
                            # stretches, remove this flag and add letterbox=False.
        mosaic=0.5,         # halved from default 1.0 — Roboflow already baked
                            # 3× augmented variants per image, so we back off
                            # Ultralytics' mosaic to avoid over-augmentation
        project=PROJECT_NAME,
        name=RUN_NAME,
    )

    # best.pt = highest val mAP checkpoint across all epochs
    return Path(results.save_dir) / "weights" / "best.pt"


def export_onnx(weights: Path) -> Path:
    model = YOLO(str(weights))
    model.export(
        format="onnx",
        imgsz=640,
        dynamic=False,      # REQUIRED: trtexec and Isaac ROS need a fixed input
                            # shape. YOLO defaults to dynamic=True, which breaks
                            # the trtexec → .plan conversion.
    )
    # Ultralytics writes best.onnx alongside best.pt
    return weights.with_suffix(".onnx")


if __name__ == "__main__":
    print("── Step 1/3: Downloading dataset from Roboflow ──")
    data_yaml = download_dataset()

    print("\n── Step 2/3: Training ──")
    best_pt = train(data_yaml)

    print("\n── Step 3/3: Exporting to ONNX ──")
    onnx_path = export_onnx(best_pt)

    print("\n── Done ─────────────────────────────────────────────────────────")
    print(f"  Weights : {best_pt}")
    print(f"  ONNX    : {onnx_path}")
    print()
    print("Next steps on the Jetson:")
    print(f"  1. Copy {onnx_path.name} to the Jetson")
    print("  2. Run trtexec to compile the TensorRT engine:")
    print(f"       /usr/src/tensorrt/bin/trtexec \\")
    print(f"           --onnx={onnx_path.name} \\")
    print(f"           --saveEngine=best.plan \\")
    print(f"           --fp16")
    print("  3. Use best.plan as engine_file_path in your isaac_ros_yolov8 launch")
    print("─────────────────────────────────────────────────────────────────────")