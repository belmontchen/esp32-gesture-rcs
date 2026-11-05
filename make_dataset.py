"""Extract frames at 5 fps and assign segment labels (rock/paper/scissors).

Labels come from manual inspection of 1 fps contact sheets (2026-11-04/05).
Segments are (start_s, end_s) ranges inside which the gesture is stable;
frames too close to a boundary are dropped.
"""
import glob
import os
import subprocess

import imageio_ffmpeg

SRC = {
    "1a51c2ee": "D:/素材/1a51c2eec47dd9a1f2ec2f0eecc7ee72.mp4",
    "803d710c": "D:/素材/803d710c2876e253147a115e693d8513_raw.mp4",
    "9d4900cf": "D:/素材/9d4900cf2cae25a87c9a8c76a05283da.mp4",
    "c314e006": "D:/素材/c314e006cad4f84addcf7407359097bc_raw.mp4",
    "32777379": "D:/素材/32777379facd5f41462165dc9a8d9a64.mp4",
    "359b5063": "D:/素材/359b5063482e204b43776e313f87f9fc_raw.mp4",
    "a928747a": "D:/素材/a928747abe18e662d1bfdada78aa3e88.mp4",
    "c0cfc796": "D:/素材/c0cfc7968dfcda2aac37dd370c3914ea.mp4",
}

SEGMENTS = {
    "1a51c2ee": [(0.0, 0.85, "rock"), (0.95, 2.75, "scissors"), (2.90, 3.85, "rock"),
                 (5.00, 5.85, "paper"), (6.00, 6.80, "rock"), (6.95, 7.75, "scissors"),
                 (7.95, 9.75, "rock"), (10.00, 10.55, "paper")],
    "803d710c": [(0.0, 1.85, "scissors"), (3.05, 6.75, "scissors"), (8.05, 10.68, "scissors")],
    "9d4900cf": [(0.0, 18.50, "rock")],
    "c314e006": [(0.0, 1.85, "scissors"), (3.05, 4.85, "scissors"), (7.05, 7.82, "scissors")],
    # 2026-11-05 新增（白墙背景、深色袖子）
    "32777379": [(0.0, 22.30, "paper")],
    "359b5063": [(0.0, 10.30, "scissors")],
    "a928747a": [(0.70, 13.30, "scissors")],
    "c0cfc796": [(0.0, 14.10, "paper")],
}

FPS = 5
OUT = r"D:\esp32-gesture\data\frames5"


def main():
    exe = imageio_ffmpeg.get_ffmpeg_exe()
    total = 0
    for tag, path in SRC.items():
        d = os.path.join(OUT, tag)
        os.makedirs(d, exist_ok=True)
        for f in glob.glob(os.path.join(d, "*.jpg")):
            os.remove(f)
        subprocess.run([exe, "-y", "-i", path, "-vf", f"fps={FPS}", "-q:v", "2",
                        os.path.join(d, "f_%04d.jpg")], capture_output=True, check=False)
        n = len(glob.glob(os.path.join(d, "*.jpg")))
        total += n
        print(tag, n, "frames")
    print("total", total)


if __name__ == "__main__":
    main()
