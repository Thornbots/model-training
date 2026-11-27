"""Independently project camera-frame ground truth and check YOLO pose labels."""
import argparse
import json
import math
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('dataset', type=Path)
args = parser.parse_args()
checked = 0
images = sorted((args.dataset / 'truth').glob('*.json'))
assert images, 'No ground-truth sidecars found'
for path in images:
    truth = json.loads(path.read_text())
    width, height = truth['width'], truth['height']
    focal = width / (2 * math.tan(math.radians(truth['hfov_degrees']) / 2))
    lines = (args.dataset / 'labels' / (path.stem + '.txt')).read_text().splitlines()
    assert len(lines) == len(truth['panels']), path
    for line, panel in zip(lines, truth['panels']):
        fields = list(map(float, line.split()))
        assert len(fields) == 17 and int(fields[0]) == panel['class_id'], path
        for i, (x, y, z) in enumerate(panel['keypoints_camera']):
            assert z > 0, path
            px, py = width / 2 + focal * x / z, height / 2 + focal * y / z
            lx, ly, visibility = fields[5 + 3*i:8 + 3*i]
            error = math.hypot(px - lx * width, py - ly * height)
            assert error <= 0.5, f'{path}: keypoint {i}: {error:.3f} px'
            assert visibility == panel['visibility'][i], path
            checked += 1
print(f'Checked {len(images)} images, {checked // 4} panels, {checked} keypoints: all within 0.5 px')
