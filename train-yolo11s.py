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

import argparse
import re
from getpass import getpass
from pathlib import Path

from roboflow import Roboflow
from ultralytics import YOLO


# ── CONFIG ────────────────────────────────────────────────────────────────────
ROBOFLOW_WORKSPACE = "rhitcv"               # shown in your Roboflow URL
ROBOFLOW_PROJECT   = "icon-detection-test"  # project slug, not display name

PROJECT_NAME = "yolo11s_realsense"          # groups runs under runs/<PROJECT_NAME>/
# ─────────────────────────────────────────────────────────────────────────────


def prompt_roboflow_version() -> int:
    """Prompt the user for the Roboflow dataset version number."""
    while True:
        raw = input("Roboflow dataset version (integer): ").strip()
        if re.fullmatch(r"\d+", raw):
            return int(raw)
        print("  Please enter a positive integer (e.g. 4).")


def next_run_name(project_name: str) -> str:
    """
    Scan runs/<project_name>/ for existing vN directories and return the
    next one (e.g. if v1 and v2 exist, return 'v3').
    Returns 'v1' if no prior runs are found.
    """
    runs_root = Path("runs") / project_name
    if not runs_root.exists():
        return "v1"

    existing = []
    for d in runs_root.iterdir():
        m = re.fullmatch(r"v(\d+)", d.name)
        if m and d.is_dir():
            existing.append(int(m.group(1)))

    return f"v{max(existing) + 1}" if existing else "v1"


def download_dataset(version: int) -> Path:
    api_key = getpass("Roboflow API key: ")
    rf = Roboflow(api_key=api_key)
    dataset = (
        rf.workspace(ROBOFLOW_WORKSPACE)
          .project(ROBOFLOW_PROJECT)
          .version(version)
          .download("yolov11")
    )
    return Path(dataset.location) / "data.yaml"


def train(data_yaml: Path, run_name: str) -> Path:
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
        name=run_name,
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


# ── ERROR VISUALISATION ───────────────────────────────────────────────────────

def visualize_errors(
    weights: Path,
    splits: list,
    data_yaml: Path,
    conf: float = 0.25,
    iou_threshold: float = 0.45,
    max_images: int = 50,
) -> None:
    """
    Run inference on the requested dataset splits and save annotated images of
    samples where the model made an incorrect detection (false positive,
    false negative, or wrong class).

    Images are written to <run_dir>/error_viz/<split>/ alongside the weights dir.

    Annotation colour key
    ─────────────────────
      Green  box labelled GT:…   → ground-truth box the model correctly found (TP)
      Red    box labelled GT:…   → ground-truth box the model missed (FN)
      Orange box labelled P:…    → model prediction with no matching GT (FP)

    Args:
        weights:       Path to best.pt (or any .pt checkpoint).
        splits:        List of split names to evaluate, e.g. ["train", "val", "test"].
        data_yaml:     Path to the dataset data.yaml so we can find images/labels.
        conf:          Confidence threshold for predictions.
        iou_threshold: IoU threshold used when matching predictions to ground truth.
        max_images:    Cap on how many error images to save per split.
    """
    import yaml
    import cv2

    save_root = weights.parent.parent / "error_viz"
    model = YOLO(str(weights))

    # Load class names from data.yaml
    with open(data_yaml) as f:
        data_cfg = yaml.safe_load(f)
    class_names = data_cfg.get("names", {})
    if isinstance(class_names, list):
        class_names = {i: n for i, n in enumerate(class_names)}

    dataset_root = data_yaml.parent

    # Map split name → image / label directories (Roboflow layout)
    split_dir_map = {
        "train": dataset_root / "train" / "images",
        "val":   dataset_root / "valid" / "images",
        "valid": dataset_root / "valid" / "images",
        "test":  dataset_root / "test"  / "images",
    }
    label_dir_map = {
        "train": dataset_root / "train" / "labels",
        "val":   dataset_root / "valid" / "labels",
        "valid": dataset_root / "valid" / "labels",
        "test":  dataset_root / "test"  / "labels",
    }

    IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    for split in splits:
        split_key = split.lower()
        img_dir = split_dir_map.get(split_key)
        lbl_dir = label_dir_map.get(split_key)

        if img_dir is None or not img_dir.exists():
            print(f"  [error_viz] Skipping '{split}': image directory not found ({img_dir})")
            continue

        out_dir = save_root / split_key
        out_dir.mkdir(parents=True, exist_ok=True)

        image_paths = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXTS)
        if not image_paths:
            print(f"  [error_viz] No images found in {img_dir}")
            continue

        print(f"  [error_viz] Evaluating {len(image_paths)} '{split}' images …")

        saved = 0
        for img_path in image_paths:
            if saved >= max_images:
                break

            # ── Ground-truth boxes (YOLO format: cls cx cy w h, normalised) ─
            lbl_path = (lbl_dir / img_path.stem).with_suffix(".txt") if lbl_dir else None
            gt_boxes = []  # (cls_id, cx, cy, w, h)  – normalised
            if lbl_path and lbl_path.exists():
                for line in lbl_path.read_text().splitlines():
                    parts = line.strip().split()
                    if len(parts) == 5:
                        gt_boxes.append((int(parts[0]), *map(float, parts[1:])))

            # ── Run inference ────────────────────────────────────────────────
            result = model(str(img_path), conf=conf, verbose=False)[0]
            img_h, img_w = result.orig_shape

            def norm_to_abs(cx, cy, w, h):
                return (
                    (cx - w / 2) * img_w, (cy - h / 2) * img_h,
                    (cx + w / 2) * img_w, (cy + h / 2) * img_h,
                )

            def iou(a, b):
                ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
                ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
                inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                if inter == 0:
                    return 0.0
                return inter / ((a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter)

            gt_abs  = [(cls, norm_to_abs(cx, cy, w, h)) for cls, cx, cy, w, h in gt_boxes]
            pred_abs = []
            if result.boxes is not None and len(result.boxes):
                for box in result.boxes:
                    pred_abs.append((
                        int(box.cls[0].item()),
                        float(box.conf[0].item()),
                        tuple(box.xyxy[0].tolist()),
                    ))

            # ── Greedy GT↔pred matching ──────────────────────────────────────
            gt_matched   = [False] * len(gt_abs)
            pred_matched = [False] * len(pred_abs)

            for gi, (gt_cls, gt_box) in enumerate(gt_abs):
                best_iou, best_pi = 0.0, -1
                for pi, (pred_cls, _, pred_box) in enumerate(pred_abs):
                    if pred_matched[pi]:
                        continue
                    ov = iou(gt_box, pred_box)
                    if ov > best_iou:
                        best_iou, best_pi = ov, pi
                if best_pi >= 0 and best_iou >= iou_threshold:
                    if pred_abs[best_pi][0] == gt_cls:   # correct class
                        gt_matched[gi]       = True
                        pred_matched[best_pi] = True
                    # class mismatch → both stay unmatched (FP + FN)

            has_error = any(not m for m in gt_matched) or any(not m for m in pred_matched)
            if not has_error:
                continue

            # ── Draw annotations ─────────────────────────────────────────────
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            annotated = img.copy()

            C = {
                "tp_gt":  (0, 200,   0),   # green  – matched GT
                "fn_gt":  (0,   0, 220),   # red    – missed GT
                "fp":     (0, 165, 255),   # orange – false positive pred
            }
            FONT = cv2.FONT_HERSHEY_SIMPLEX

            for gi, (gt_cls, (x1, y1, x2, y2)) in enumerate(gt_abs):
                color = C["tp_gt"] if gt_matched[gi] else C["fn_gt"]
                label = f"GT:{class_names.get(gt_cls, gt_cls)}"
                cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                cv2.putText(annotated, label, (int(x1), max(int(y1)-6, 10)),
                            FONT, 0.5, color, 1, cv2.LINE_AA)

            for pi, (pred_cls, pred_conf, (x1, y1, x2, y2)) in enumerate(pred_abs):
                if pred_matched[pi]:
                    continue  # TP predictions already shown via GT box
                label = f"P:{class_names.get(pred_cls, pred_cls)} {pred_conf:.2f}"
                cv2.rectangle(annotated, (int(x1)+1, int(y1)+1), (int(x2)+1, int(y2)+1),
                              C["fp"], 2)
                cv2.putText(annotated, label, (int(x1)+1, max(int(y1)-6, 20)),
                            FONT, 0.5, C["fp"], 1, cv2.LINE_AA)

            # Small legend
            legend = [("GT matched", C["tp_gt"]), ("GT missed (FN)", C["fn_gt"]),
                      ("False positive", C["fp"])]
            for li, (txt, col) in enumerate(legend):
                y = 18 + li * 18
                cv2.rectangle(annotated, (6, y-12), (16, y), col, -1)
                cv2.putText(annotated, txt, (20, y), FONT, 0.45, (255,255,255), 1, cv2.LINE_AA)

            cv2.imwrite(str(out_dir / img_path.name), annotated)
            saved += 1

        print(f"  [error_viz] Saved {saved} error image(s) → {out_dir}")

    print(f"  [error_viz] Complete. All outputs under: {save_root}")


# ── CLI ENTRY POINT ───────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="YOLO11s training pipeline with optional post-run error visualisation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples
--------
  # Normal training run (prompts for Roboflow version; val error viz runs automatically)
  python train-yolo11s.py

  # Training + error viz on all three splits
  python train-yolo11s.py --viz-splits train val test

  # Training only, skip error viz
  python train-yolo11s.py --no-viz

  # Error visualisation on an existing run (no training)
  python train-yolo11s.py --viz-only runs/yolo11s_realsense/v3/weights/best.pt --viz-splits val test

  # Same, but you also need to point at the dataset (if it moved)
  python train-yolo11s.py --viz-only best.pt --data-yaml path/to/data.yaml
""",
    )
    p.add_argument(
        "--viz-only", metavar="WEIGHTS",
        help="Skip training; run error visualisation on existing weights (path to best.pt).",
    )
    p.add_argument(
        "--viz-splits", nargs="+", default=["val"],
        metavar="SPLIT",
        choices=["train", "val", "valid", "test"],
        help="Which splits to visualise (default: val). Choices: train val test.",
    )
    p.add_argument(
        "--viz-conf", type=float, default=0.25,
        help="Confidence threshold for error visualisation (default: 0.25).",
    )
    p.add_argument(
        "--viz-iou", type=float, default=0.45,
        help="IoU threshold for GT↔pred matching (default: 0.45).",
    )
    p.add_argument(
        "--viz-max", type=int, default=50,
        help="Max error images to save per split (default: 50).",
    )
    p.add_argument(
        "--no-viz", action="store_true",
        help="Skip automatic post-training error visualisation.",
    )
    p.add_argument(
        "--data-yaml", metavar="PATH",
        help="Path to data.yaml (only needed with --viz-only if the dataset moved).",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # ── Visualisation-only mode ───────────────────────────────────────────────
    if args.viz_only:
        weights = Path(args.viz_only)
        if not weights.exists():
            raise FileNotFoundError(f"Weights not found: {weights}")

        if args.data_yaml:
            data_yaml = Path(args.data_yaml)
        else:
            candidates = list(Path(".").rglob("data.yaml"))
            if not candidates:
                raise RuntimeError(
                    "Could not locate data.yaml automatically. "
                    "Pass --data-yaml <path> explicitly."
                )
            data_yaml = candidates[0]
            print(f"  Using data.yaml: {data_yaml}")

        print("── Error Visualisation ──────────────────────────────────────────")
        visualize_errors(
            weights=weights,
            splits=args.viz_splits,
            data_yaml=data_yaml,
            conf=args.viz_conf,
            iou_threshold=args.viz_iou,
            max_images=args.viz_max,
        )
        raise SystemExit(0)

    # ── Full training pipeline ────────────────────────────────────────────────
    roboflow_version = prompt_roboflow_version()
    run_name = next_run_name(PROJECT_NAME)
    print(f"  Dataset version : {roboflow_version}")
    print(f"  Run name        : {run_name}")

    print("\n── Step 1/3: Downloading dataset from Roboflow ──")
    data_yaml = download_dataset(roboflow_version)

    print("\n── Step 2/3: Training ──")
    best_pt = train(data_yaml, run_name)

    print("\n── Step 3/3: Exporting to ONNX ──")
    onnx_path = export_onnx(best_pt)

    if not args.no_viz:
        print(f"\n── Post-training Error Visualisation (splits: {args.viz_splits}) ──")
        print("  (pass --no-viz to skip, or --viz-splits train val test to change splits)")
        visualize_errors(
            weights=best_pt,
            splits=args.viz_splits,
            data_yaml=data_yaml,
            conf=args.viz_conf,
            iou_threshold=args.viz_iou,
            max_images=args.viz_max,
        )

    print("\n── Done ─────────────────────────────────────────────────────────")
    print(f"  Weights : {best_pt}")
    print(f"  ONNX    : {onnx_path}")
    print()
    print("Next steps on the Jetson:")
    print(f"  1. Copy {onnx_path.name} to the Jetson")
    print("  2. Enter Isaac ROS")
    print("  3. Run trtexec to compile the TensorRT engine:")
    print(f"       /usr/src/tensorrt/bin/trtexec \\")
    print(f"           --onnx={onnx_path.name} \\")
    print(f"           --saveEngine=${{ISAAC_ROS_WS}}/isaac_ros_assets/models/yolo11/yolo11s_fp16.plan\\")
    print(f"           --fp16")
    print("  4. Use best.plan as engine_file_path in your isaac_ros_yolov8 launch")
    print("─────────────────────────────────────────────────────────────────────")