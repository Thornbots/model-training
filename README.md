# model-training

# *WARNING*
*YOU MUST INSTALL GIT-LFS before cloning*

then clone by 
```bash
git clone --recurse-submodules https://github.com/Thornbots/model-training.git
```


## Git submodules are hard, example of updating them below:
``` bash
cd runs
git add .
git commit -m "yolo11s v1 - mAP50 0.91"
git push
cd ..
git add runs          # updates the submodule pointer (commit hash) in the main repo
git commit -m "update models to v1"
git push
```

## Setup and usage (YOLO26)

``` bash
python3.12 -m venv ~/venvs/yolo26 && . ~/venvs/yolo26/bin/activate   # keep the venv outside the repo
pip install -r requirements.txt

python train-yolo26.py --smoke                 # synthetic data, 1 epoch, export + shape check (no data/Roboflow)
python train-yolo26.py --project <slug> --version <N>   # pose (default), prompts for the API key
python train-yolo26.py --task detect --project icon-detection-test --version <N>   # today's boxes
python train-yolo26.py --size m --epochs 200 --batch 32 --device 0
python train-yolo26.py --viz-only runs/<run>/v1/weights/best.pt --task detect --data-yaml <path>
```

`--task pose` (default) trains `yolo26<size>-pose` with `kpt_shape [4, 3]`, `flip_idx [1, 0, 3, 2]`
(patched into `data.yaml` if Roboflow omits them). The Roboflow project is a
placeholder (`ROBOFLOW_PROJECT` in the script) until the keypoint project exists.
Box error visualisation works for `--task detect` only.

### Export contract

Static 640x640 ONNX, NMS-free end2end head (`nms=False`, `dynamic=False`), input `images`
`[1,3,640,640]`, output `output0`:

| task | output0 | row |
| --- | --- | --- |
| pose | `[1, 300, 18]` | `x1 y1 x2 y2 score class` then `(kx ky kconf) x4`, network-input pixels |
| detect | `[1, 300, 6]` | `x1 y1 x2 y2 score class` |

The script verifies this with onnxruntime after every export and raises on mismatch.
Compile on the Jetson with the `trtexec` command it prints. Verified with Ultralytics 8.4.220
(`requirements.txt`); re-run `--smoke` before bumping the pin.

## Dataset and labelling spec (armor keypoints)

Panel geometry is not final (no physical panel or CAD yet). The decisions below are
placeholders to revisit when it exists.

- **One instance per armor panel** with a box plus 4 keypoints, the light-bar ends as seen
  in the image, in this order: **0 TL, 1 TR, 2 BR, 3 BL** (top of the image is "top"; do
  not order by panel orientation). Left/right follow the image, so a horizontal flip
  swaps TL<->TR and BR<->BL (`flip_idx [1,0,3,2]`). Keep the same four physical points
  on every panel, whichever side faces the camera.
- **Visibility** (Roboflow/YOLO `v`): 2 = visible, 1 = occluded but the position is
  confidently inferable (label it where it is), 0 = unknown (leave unlabelled, x=y=0).
  Never guess a position for `v=0`. Keep partly occluded panels rather than dropping them.
- **Box** tightly encloses the four keypoints plus the light bars.
- **Classes** (placeholder, matches today's 8-class model and the
  [`target_selector`](https://github.com/Thornbots/isaac_ros-dev/blob/nightly/thornbots_pkg/README.md#target_selector)
  contract): ids 0-3 blue, 4-7 red. Fix the id-to-robot mapping before labelling; the
  selector drops 0-3 on the blue team and 4-7 on red.
- **Capture**: RealSense colour 640x480 with the locked manual exposure from the
  `realsense-yolov8-nitros-bridge` `cv/manual-color-exposure` branch
  [Camera profile](https://github.com/Thornbots/realsense-yolov8-nitros-bridge/blob/cv/manual-color-exposure/README.md#camera-profile)
  (auto-exposure off, short exposure so spinning armor stays sharp). Train on frames
  captured with the same settings. Do not rescale or pad images by hand: Ultralytics
  letterboxes to 640x640 as the Isaac ROS encoder does, so keypoints stay consistent.
- Cover distance, angle (including steep oblique views, which the pose head is for),
  spin, blur, lighting and both team colours; hold out whole sessions for val/test.

## Legacy (YOLO11)

`train-yolo11s.py` was replaced by `train-yolo26.py --task detect`.

## Synthetic armor sample

The headless Blender generator uses the MIT-licensed
[armor_module CAD and decals](https://github.com/jiushipanta/armor_module).
Its class mapping is a **placeholder**: blue Hero (decal 1), Engineer (2),
Infantry (3 or 4), Sentry are IDs 0–3; red equivalents are IDs 4–7.
It mounts AM02 panels with sticker or procedural dot-matrix faces on sentry
CAD using the sim URDF armor joints. Scenes use the ARC field meshes, posed
robot parts, team-colour indicators, and bright emissive armor; one quarter
use HDR backgrounds. Most robots are 1.5–4 m away, with 1–6 m coverage.
`--heldout` selects a separate HDR background, dot grid and lighting ranges.
These scenes are approximations, not measured RealSense imagery.
The JSON sidecars record camera-frame position, outward normal and light-bar
endpoints in metres (x right, y down, z forward), plus class, colour and face.
The label checker independently projects those endpoints to compare with the
YOLO pose labels. This is a small sample, not a training dataset or a substitute
for validating against physical panels.

Use the sim checkout containing `urdf/sentry_v2` and `world/arcc_s2_3v3`,
and a directory with at least two `.hdr` files, such as CC0 maps from
[Poly Haven](https://polyhaven.com/hdris). The first and last sorted maps are
reserved for training and held-out scenes respectively. For example, from this
repository on macOS with Blender 4.3.2 installed:

```bash
git clone https://github.com/jiushipanta/armor_module.git /tmp/armor_module
python3.12 -m venv /tmp/armor-synthetic-venv
/tmp/armor-synthetic-venv/bin/pip install numpy opencv-python-headless
/path/to/Blender.app/Contents/MacOS/Blender -b -t 8 --python synthetic/render.py -- --assets /tmp/armor_module --sim /path/to/sim --hdri /path/to/hdris --output /tmp/armor-synthetic-sample --count 40 --seed 42
/tmp/armor-synthetic-venv/bin/python synthetic/sensor_noise.py /tmp/armor-synthetic-sample/images --seed 42
/tmp/armor-synthetic-venv/bin/python synthetic/check_labels.py /tmp/armor-synthetic-sample
/tmp/armor-synthetic-venv/bin/python synthetic/overlay_labels.py /tmp/armor-synthetic-sample /tmp/armor-synthetic-overlays
```

The noise command modifies the rendered JPEGs in place; rerender before running
it again. The overlay command writes the first six nonempty examples. Keep
rendered datasets outside git, and set `synthetic/config/data.yaml`'s `path` to
a suitably split dataset before training.
