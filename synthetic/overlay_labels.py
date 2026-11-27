"""Draw the first six nonempty YOLO pose labels over their source images."""
import argparse
from pathlib import Path

import cv2

NAMES = ('blue_hero', 'blue_engineer', 'blue_infantry', 'blue_sentry',
         'red_hero', 'red_engineer', 'red_infantry', 'red_sentry')
parser = argparse.ArgumentParser()
parser.add_argument('dataset', type=Path)
parser.add_argument('output', type=Path)
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=True)
written = 0
for path in sorted((args.dataset / 'labels').glob('*.txt')):
    lines = path.read_text().splitlines()
    if not lines:
        continue
    image = cv2.imread(str(args.dataset / 'images' / (path.stem + '.jpg')))
    if image is None:
        raise FileNotFoundError(path.stem)
    height, width = image.shape[:2]
    for line in lines:
        fields = list(map(float, line.split()))
        cls, cx, cy, bw, bh = fields[:5]
        p1 = (round((cx - bw/2)*width), round((cy - bh/2)*height))
        p2 = (round((cx + bw/2)*width), round((cy + bh/2)*height))
        cv2.rectangle(image, p1, p2, (0, 255, 0), 1)
        cv2.putText(image, NAMES[int(cls)], (p1[0], max(10, p1[1]-3)),
                    cv2.FONT_HERSHEY_SIMPLEX, .35, (0, 255, 0), 1)
        for i in range(4):
            x, y, visibility = fields[5+3*i:8+3*i]
            if visibility:
                point = (round(x*width), round(y*height))
                cv2.circle(image, point, 2, (0, 255, 255), -1)
                cv2.putText(image, str(i), (point[0]+3, point[1]-2),
                            cv2.FONT_HERSHEY_SIMPLEX, .35, (0, 255, 255), 1)
    target = args.output / (path.stem + '.png')
    if not cv2.imwrite(str(target), image):
        raise OSError(target)
    print(target)
    written += 1
    if written == 6:
        break
if written < 6:
    raise ValueError(f'Only {written} nonempty images found')
