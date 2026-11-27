"""Apply reproducible gain, sensor noise and JPEG compression to generated frames."""
import argparse
from pathlib import Path

import cv2
import numpy as np

p = argparse.ArgumentParser()
p.add_argument('images', type=Path)
p.add_argument('--seed', type=int, required=True)
a = p.parse_args()
rng = np.random.default_rng(a.seed)
for path in sorted(a.images.glob('*.jpg')):
    image = cv2.imread(str(path)).astype(np.float32)
    image = image * rng.uniform(.8, 1.2) + rng.normal(0, rng.uniform(.5, 5), image.shape)
    image = np.clip(image, 0, 255).astype(np.uint8)
    if not cv2.imwrite(str(path), image, [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(55, 99))]):
        raise OSError(path)
