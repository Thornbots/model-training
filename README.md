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

## Usage of this script:
``` bash
# Normal — prompts for version, auto-names run, runs val error viz after training
python train-yolo11s.py

# Error viz on all splits after training
python train-yolo11s.py --viz-splits train val test

# Skip viz entirely
python train-yolo11s.py --no-viz

# Retroactively run viz on a finished run (no retraining)
python train-yolo11s.py --viz-only runs/yolo11s_realsense/v2/weights/best.pt --viz-splits val test

# Tune thresholds (lower conf catches more FPs; higher iou is stricter matching)
python train-yolo11s.py --viz-conf 0.15 --viz-iou 0.5 --viz-max 100
```
