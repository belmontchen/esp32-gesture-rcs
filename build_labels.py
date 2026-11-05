"""Assign segment labels to 5fps frames and write data/dataset.csv."""
import csv
import glob
import os

from make_dataset import SEGMENTS, FPS, OUT

MARGIN = 0.15


def label_for(t, segs):
    for a, b, lab in segs:
        if a + MARGIN <= t <= b - MARGIN:
            return lab
    return None


def main():
    rows = []
    for tag, segs in SEGMENTS.items():
        for f in sorted(glob.glob(os.path.join(OUT, tag, "*.jpg"))):
            idx = int(os.path.basename(f).split("_")[1].split(".")[0])
            t = (idx - 1) / FPS
            lab = label_for(t, segs)
            if lab:
                rows.append({"file": f, "video": tag, "time": round(t, 2), "label": lab})
    out = r"D:\esp32-gesture\data\dataset.csv"
    with open(out, "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=["file", "video", "time", "label"])
        wr.writeheader()
        wr.writerows(rows)
    from collections import Counter
    print("rows", len(rows), Counter(r["label"] for r in rows))
    per = Counter((r["video"], r["label"]) for r in rows)
    for k, v in sorted(per.items()):
        print(k, v)
    print("wrote", out)


if __name__ == "__main__":
    main()
