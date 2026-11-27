#!/usr/bin/env python3
"""YOLO26 armor training: Roboflow download -> train -> NMS-free ONNX export.

Export contract (static 640x640, end2end, see README):
  pose   output0 [1, 300, 6 + 3*K]  (K=4: TL,TR,BR,BL light-bar ends)
  detect output0 [1, 300, 6]
The export is verified and fails loudly on any other shape.
`--smoke` runs the whole chain on a synthetic dataset (no Roboflow, no data).
"""

import argparse
import re
import shutil
import tempfile
from getpass import getpass
from pathlib import Path

import yaml

# ── CONFIG (placeholders: the new keypoint Roboflow project does not exist yet)
ROBOFLOW_WORKSPACE = "rhitcv"
ROBOFLOW_PROJECT = "armor-keypoints-placeholder"   # TODO: real project slug
ROBOFLOW_FORMAT = "yolov11"                        # Roboflow keypoint export
LEGACY_DETECT_PROJECT = "icon-detection-test"      # today's detect dataset

KPT_SHAPE = [4, 3]        # 4 keypoints x (x, y, visibility)
FLIP_IDX = [1, 0, 3, 2]   # hflip swaps TL<->TR and BR<->BL
IMGSZ = 640
MAX_DET = 300             # end2end head keeps the top 300 rows
# ─────────────────────────────────────────────────────────────────────────────


def prompt_roboflow_version() -> int:
    while True:
        raw = input("Roboflow dataset version (integer): ").strip()
        if re.fullmatch(r"\d+", raw):
            return int(raw)
        print("  Please enter a positive integer (e.g. 4).")


def next_run_name(project_name: str) -> str:
    """Next vN under runs/<project_name>/ (v1 if none)."""
    runs_root = Path("runs") / project_name
    nums = [int(m.group(1)) for d in runs_root.iterdir()
            if (m := re.fullmatch(r"v(\d+)", d.name)) and d.is_dir()] \
        if runs_root.exists() else []
    return f"v{max(nums) + 1}" if nums else "v1"


def download_dataset(workspace: str, project: str, version: int, task: str) -> Path:
    from roboflow import Roboflow
    rf = Roboflow(api_key=getpass("Roboflow API key: "))
    ds = rf.workspace(workspace).project(project).version(version).download(ROBOFLOW_FORMAT)
    data_yaml = Path(ds.location) / "data.yaml"
    if task == "pose":
        ensure_pose_keys(data_yaml)
    return data_yaml


def ensure_pose_keys(data_yaml: Path) -> None:
    """Roboflow's data.yaml can omit kpt_shape/flip_idx; flip_idx is not exported."""
    cfg = yaml.safe_load(data_yaml.read_text())
    changed = False
    if cfg.get("kpt_shape") != KPT_SHAPE:
        if "kpt_shape" in cfg:
            print(f"  [warn] data.yaml kpt_shape {cfg['kpt_shape']} -> {KPT_SHAPE}")
        cfg["kpt_shape"] = KPT_SHAPE
        changed = True
    if cfg.get("flip_idx") != FLIP_IDX:
        cfg["flip_idx"] = FLIP_IDX
        changed = True
    if changed:
        data_yaml.write_text(yaml.safe_dump(cfg, sort_keys=False))


def train(data_yaml: Path, args, run_name: str) -> Path:
    from ultralytics import YOLO
    suffix = "-pose" if args.task == "pose" else ""
    # .yaml = random init (smoke); .pt = pretrained (head remaps kpt count)
    model = YOLO(args.model or f"yolo26{args.size}{suffix}" + (".yaml" if args.scratch else ".pt"))
    kw = dict(
        data=str(data_yaml), epochs=args.epochs, patience=args.patience,
        imgsz=args.imgsz, batch=args.batch, device=args.device,
        project=args.project_dir, name=run_name, exist_ok=True,
        plots=not args.smoke, workers=0 if args.smoke else 8,
    )
    if not args.smoke:
        # rect + halved mosaic carried over from the YOLO11 flow (Roboflow already augments).
        kw.update(rect=True, mosaic=0.5)
    if args.task == "pose":
        kw["fliplr"] = 0.5     # safe only because flip_idx swaps the corners
    results = model.train(**kw)
    return Path(results.save_dir) / "weights" / "best.pt"


def export_onnx(weights: Path, task: str, imgsz: int) -> Path:
    from ultralytics import YOLO
    out = YOLO(str(weights)).export(
        format="onnx", imgsz=imgsz,
        dynamic=False,   # trtexec/Isaac ROS need a fixed input shape
        nms=False,       # YOLO26 default: NMS-free one-to-one end2end head
        simplify=True, batch=1,
    )
    onnx_path = Path(out)
    verify_onnx(onnx_path, task, imgsz)
    return onnx_path


def verify_onnx(onnx_path: Path, task: str, imgsz: int) -> None:
    """Fail loudly unless input/output match the decoder contract."""
    import numpy as np
    import onnxruntime as ort
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    inp, outs = sess.get_inputs(), sess.get_outputs()
    in_ok = len(inp) == 1 and inp[0].name == "images" and list(inp[0].shape) == [1, 3, imgsz, imgsz]
    cols = 6 + 3 * KPT_SHAPE[0] if task == "pose" else 6
    want = [1, MAX_DET, cols]
    out_ok = len(outs) == 1 and outs[0].name == "output0" and list(outs[0].shape) == want
    print(f"  [verify] input : {[(i.name, i.shape) for i in inp]}")
    print(f"  [verify] output: {[(o.name, o.shape) for o in outs]}")
    got = sess.run(None, {inp[0].name: np.zeros([1, 3, imgsz, imgsz], np.float32)})
    print(f"  [verify] runtime output shape: {[g.shape for g in got]}")
    if not (in_ok and out_ok and list(got[0].shape) == want):
        raise RuntimeError(
            f"ONNX contract mismatch for {onnx_path}: expected input images "
            f"[1,3,{imgsz},{imgsz}] and output0 {want}")
    print(f"  [verify] OK: images -> output0 {want}")


# ── SMOKE DATASET ─────────────────────────────────────────────────────────────

def make_synthetic_dataset(root: Path, task: str, n_train: int = 12, n_val: int = 4) -> Path:
    """Rotated light rectangles on noise; corners ordered TL,TR,BR,BL, visibility 2."""
    import cv2
    import numpy as np
    rng = np.random.default_rng(0)
    W, H = 640, 480
    for split, n in (("train", n_train), ("val", n_val)):
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)
        for i in range(n):
            img = rng.integers(0, 80, (H, W, 3), dtype=np.uint8)
            w, h = rng.uniform(80, 200), rng.uniform(40, 90)
            cx, cy = rng.uniform(150, W - 150), rng.uniform(100, H - 100)
            a = np.deg2rad(rng.uniform(-20, 20))
            R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
            local = np.array([[-w, -h], [w, -h], [w, h], [-w, h]]) / 2   # TL,TR,BR,BL
            pts = local @ R.T + [cx, cy]
            cv2.fillPoly(img, [pts.astype(np.int32)], (230, 230, 230))
            cv2.imwrite(str(root / "images" / split / f"{i}.jpg"), img)
            x0, y0 = pts.min(0)
            x1, y1 = pts.max(0)
            box = [(x0 + x1) / 2 / W, (y0 + y1) / 2 / H, (x1 - x0) / W, (y1 - y0) / H]
            if task == "pose":
                kp = [f"{x / W:.6f} {y / H:.6f} 2" for x, y in pts]
                row = f"0 {box[0]:.6f} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f} " + " ".join(kp)
            else:
                row = "0 " + " ".join(f"{b:.6f}" for b in box)
            (root / "labels" / split / f"{i}.txt").write_text(row + "\n")
    cfg = {"path": str(root), "train": "images/train", "val": "images/val", "names": {0: "armor"}}
    if task == "pose":
        cfg.update(kpt_shape=KPT_SHAPE, flip_idx=FLIP_IDX)
    data_yaml = root / "data.yaml"
    data_yaml.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return data_yaml


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
    import cv2
    from ultralytics import YOLO

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

        # Warn clearly if the label directory is missing — without labels every
        # prediction will appear as a false positive, which is misleading.
        if lbl_dir is None or not lbl_dir.exists():
            print(f"  [error_viz] WARNING: label directory not found for '{split}' ({lbl_dir})")
            print(f"  [error_viz]   GT boxes cannot be loaded — all predictions will show as FP.")
            print(f"  [error_viz]   Check that your dataset has a labels/ folder next to images/")
            lbl_dir = None  # force None so the per-image check is explicit

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
            label_stem = img_path.stem.replace("_rf_", ".rf.", 1)
            lbl_path = (lbl_dir / (img_path.stem + ".txt")) if lbl_dir else None            
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
            print(f"  {img_path.name}: {len(gt_abs)} GT, {len(pred_abs)} pred, label_found={bool(lbl_path and lbl_path.exists())}")

            # ── Greedy GT↔pred matching ──────────────────────────────────────
            gt_matched   = [False] * len(gt_abs)
            pred_matched = [False] * len(pred_abs)

            for gi, (gt_cls, gt_box) in enumerate(gt_abs):
                print(f"  GT  class={gt_cls} ({class_names.get(gt_cls, '?')})")
                best_iou, best_pi = 0.0, -1
                for pi, (pred_cls, _, pred_box) in enumerate(pred_abs):
                    print(f"  Pred class={pred_cls} ({class_names.get(pred_cls, '?')}) conf={conf:.2f}")
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
                "tp_gt":   (0, 200,   0),   # green  – GT box that was matched
                "fn_gt":   (0,   0, 220),   # red    – GT box the model missed
                "tp_pred": (0, 255, 100),   # bright green – correct prediction
                "fp":      (0, 165, 255),   # orange – false positive prediction
            }
            FONT = cv2.FONT_HERSHEY_SIMPLEX

            # Draw GT boxes (green = matched, red = missed)
            for gi, (gt_cls, (x1, y1, x2, y2)) in enumerate(gt_abs):
                color = C["tp_gt"] if gt_matched[gi] else C["fn_gt"]
                label = f"GT:{class_names.get(gt_cls, gt_cls)}"
                cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                cv2.putText(annotated, label, (int(x1), max(int(y1)-8, 10)),
                            FONT, 0.45, color, 1, cv2.LINE_AA)

            # Draw ALL prediction boxes so their localization is always visible.
            # TP preds are inset by 2px so they're distinguishable from the GT outline.
            for pi, (pred_cls, pred_conf, (x1, y1, x2, y2)) in enumerate(pred_abs):
                if pred_matched[pi]:
                    color = C["tp_pred"]
                    cv2.rectangle(annotated, (int(x1)+2, int(y1)+2), (int(x2)-2, int(y2)-2),
                                  color, 1)
                    label = f"P:{class_names.get(pred_cls, pred_cls)} {pred_conf:.2f} TP"
                else:
                    color = C["fp"]
                    cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)),
                                  color, 2)
                    label = f"P:{class_names.get(pred_cls, pred_cls)} {pred_conf:.2f} FP"
                cv2.putText(annotated, label, (int(x1), max(int(y2)+12, 20)),
                            FONT, 0.45, color, 1, cv2.LINE_AA)

            # Legend
            legend = [
                ("GT matched (TP)", C["tp_gt"]),
                ("GT missed  (FN)", C["fn_gt"]),
                ("Pred TP (inset)",  C["tp_pred"]),
                ("Pred FP",          C["fp"]),
            ]
            for li, (txt, col) in enumerate(legend):
                y = 18 + li * 18
                cv2.rectangle(annotated, (6, y-12), (16, y), col, -1)
                cv2.putText(annotated, txt, (20, y), FONT, 0.42, (255,255,255), 1, cv2.LINE_AA)

            cv2.imwrite(str(out_dir / img_path.name), annotated)
            saved += 1

        print(f"  [error_viz] Saved {saved} error image(s) → {out_dir}")

    print(f"  [error_viz] Complete. All outputs under: {save_root}")


# ── CLI ENTRY POINT ───────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="YOLO26 armor training (pose default) with NMS-free ONNX export.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples
--------
  python train-yolo26.py                       # pose, prompts for key + version
  python train-yolo26.py --task detect --project icon-detection-test
  python train-yolo26.py --smoke               # synthetic data, 1 epoch, export, verify
  python train-yolo26.py --viz-only runs/yolo26s_pose_realsense/v1/weights/best.pt \\
      --task detect --data-yaml path/to/data.yaml
""",
    )
    p.add_argument("--task", choices=["pose", "detect"], default="pose")
    p.add_argument("--size", default="s", choices=["n", "s", "m", "l", "x"])
    p.add_argument("--model", help="Override weights/yaml (default yolo26<size>[-pose].pt).")
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--patience", type=int, default=50)
    p.add_argument("--imgsz", type=int, default=IMGSZ)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--device", default=None, help="Ultralytics device (default: auto).")
    p.add_argument("--workspace", default=ROBOFLOW_WORKSPACE)
    p.add_argument("--project", default=None,
                   help=f"Roboflow project slug (default {ROBOFLOW_PROJECT}; "
                        f"{LEGACY_DETECT_PROJECT} for --task detect).")
    p.add_argument("--version", type=int, help="Roboflow version (default: prompt).")
    p.add_argument("--smoke", action="store_true",
                   help="Synthetic-data end-to-end check (CPU/MPS, 1 epoch, no Roboflow).")
    p.add_argument("--scratch", action="store_true", help="Random init from the .yaml (no download).")
    p.add_argument("--viz-only", metavar="WEIGHTS",
                   help="Skip training; error visualisation on existing weights (detect only).")
    p.add_argument("--viz-splits", nargs="+", default=["val"], metavar="SPLIT",
                   choices=["train", "val", "valid", "test"])
    p.add_argument("--viz-conf", type=float, default=0.25)
    p.add_argument("--viz-iou", type=float, default=0.45)
    p.add_argument("--viz-max", type=int, default=50)
    p.add_argument("--no-viz", action="store_true")
    p.add_argument("--data-yaml", metavar="PATH", help="Existing data.yaml (skips Roboflow).")
    return p.parse_args()


def find_data_yaml(weights: Path) -> Path:
    args_yaml = weights.parent.parent / "args.yaml"
    if args_yaml.exists():
        data_yaml = Path(yaml.safe_load(args_yaml.read_text())["data"])
        print(f"  Using data.yaml from args.yaml: {data_yaml}")
        return data_yaml
    candidates = sorted(Path(".").rglob("data.yaml"))
    if not candidates:
        raise RuntimeError("Could not locate data.yaml. Pass --data-yaml <path>.")
    if len(candidates) > 1:
        print(f"  [warning] Multiple data.yaml files found, using: {candidates[0]}")
    return candidates[0]


def run_viz(weights: Path, data_yaml: Path, args) -> None:
    if args.task != "detect":
        print("  [error_viz] Box-matching viz is detect-only; skipped for pose.")
        return
    print("── Error Visualisation ──────────────────────────────────────────")
    visualize_errors(weights=weights, splits=args.viz_splits, data_yaml=data_yaml,
                     conf=args.viz_conf, iou_threshold=args.viz_iou, max_images=args.viz_max)


def print_next_steps(onnx_path: Path, name: str) -> None:
    print("\nNext steps on the Jetson:")
    print(f"  1. Copy {onnx_path.name} to the Jetson")
    print("  2. Enter Isaac ROS and compile the TensorRT engine:")
    print("       /usr/src/tensorrt/bin/trtexec \\")
    print(f"           --onnx={onnx_path.name} \\")
    print(f"           --saveEngine=${{ISAAC_ROS_WS}}/isaac_ros_assets/models/yolo26/{name}_fp16.plan \\")
    print("           --fp16")
    print("  3. Point the engine_file_path at that plan (pose: decoder:=yolo26).")


if __name__ == "__main__":
    args = parse_args()
    if args.project is None:
        args.project = LEGACY_DETECT_PROJECT if args.task == "detect" else ROBOFLOW_PROJECT
    tag = "pose" if args.task == "pose" else "det"
    args.project_dir = f"yolo26{args.size}_{tag}_realsense"
    engine_name = f"yolo26{args.size}_{tag}"

    if args.viz_only:
        weights = Path(args.viz_only)
        if not weights.exists():
            raise FileNotFoundError(f"Weights not found: {weights}")
        run_viz(weights, Path(args.data_yaml) if args.data_yaml else find_data_yaml(weights), args)
        raise SystemExit(0)

    if args.smoke:
        tmp = Path(tempfile.mkdtemp(prefix="yolo26_smoke_"))
        args.scratch, args.epochs, args.batch = True, 1, 4
        args.size = "n" if args.size == "s" else args.size
        args.project_dir = str(tmp / "runs")
        engine_name = f"yolo26{args.size}_{tag}_smoke"
        data_yaml = make_synthetic_dataset(tmp / "data", args.task)
        run_name = "smoke"
        print(f"  Smoke workdir: {tmp}")
    else:
        run_name = next_run_name(args.project_dir)
        print(f"  Task: {args.task}  Run: {run_name}")
        if args.data_yaml:
            data_yaml = Path(args.data_yaml)
            if args.task == "pose":
                ensure_pose_keys(data_yaml)
        else:
            version = args.version or prompt_roboflow_version()
            print("\n── Step 1/3: Downloading dataset from Roboflow ──")
            data_yaml = download_dataset(args.workspace, args.project, version, args.task)

    print("\n── Step 2/3: Training ──")
    best_pt = train(data_yaml, args, run_name)

    print("\n── Step 3/3: Exporting to ONNX ──")
    onnx_path = export_onnx(best_pt, args.task, args.imgsz)

    if not args.smoke and not args.no_viz:
        run_viz(best_pt, data_yaml, args)

    print("\n── Done ─────────────────────────────────────────────────────────")
    print(f"  Weights : {best_pt}")
    print(f"  ONNX    : {onnx_path}")
    if args.smoke:
        print("  SMOKE OK")
        shutil.rmtree(tmp, ignore_errors=True)
    else:
        print_next_steps(onnx_path, engine_name)
