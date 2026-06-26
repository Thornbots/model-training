# model-training



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
