"""Auto-label gesture frames with MediaPipe hand landmarks.

Rule: count extended fingers [index, middle, ring, pinky] by comparing
wrist->tip distance against wrist->pip distance.
  rock     = 0 extended
  paper    = 4 extended
  scissors = index+middle extended, ring+pinky folded
Anything else (transitions, pointing) is dropped.
"""
import csv
import glob
import os
import sys

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

MODEL = r"D:\esp32-gesture\models\hand_landmarker.task"

base_options = mp_python.BaseOptions(model_asset_path=MODEL)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
)
detector = vision.HandLandmarker.create_from_options(options)

# landmark indices
WRIST = 0
IDX = (8, 6, 5)      # tip, pip, mcp
MID = (12, 10, 9)
RING = (16, 14, 13)
PINKY = (20, 18, 17)


def dist(a, b, w, h):
    dx = (a.x - b.x) * w
    dy = (a.y - b.y) * h
    return (dx * dx + dy * dy) ** 0.5


def extended(lms, finger, w, h):
    tip, pip, _ = finger
    return dist(lms[WRIST], lms[tip], w, h) > 1.12 * dist(lms[WRIST], lms[pip], w, h)


def classify(lms, w, h):
    e = [extended(lms, f, w, h) for f in (IDX, MID, RING, PINKY)]
    n = sum(e)
    if n == 0:
        return "rock", e
    if n == 4:
        return "paper", e
    if e[0] and e[1] and not e[2] and not e[3]:
        return "scissors", e
    return "other", e


def main():
    roots = sorted(glob.glob(sys.argv[1] if len(sys.argv) > 1 else r"D:\esp32-gesture\data\raw_frames\*"))
    rows = []
    for root in roots:
        tag = os.path.basename(root)
        files = sorted(glob.glob(os.path.join(root, "*.jpg")))
        counts = {}
        for f in files:
            img = mp.Image.create_from_file(f)
            w, h = img.width, img.height
            res = detector.detect(img)
            if not res.hand_landmarks:
                label, e = "nohand", []
            else:
                label, e = classify(res.hand_landmarks[0], w, h)
            counts[label] = counts.get(label, 0) + 1
            rows.append({"file": f, "video": tag, "label": label, "fingers": "".join("1" if x else "0" for x in e)})
        print(tag, counts, flush=True)

    out = r"D:\esp32-gesture\data\labels.csv"
    with open(out, "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=["file", "video", "label", "fingers"])
        wr.writeheader()
        wr.writerows(rows)
    print("wrote", out, len(rows))


if __name__ == "__main__":
    main()
