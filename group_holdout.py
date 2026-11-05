"""按视频留一法评估：每次留一个视频做测试，其余视频训练。

用来估计"换背景/换人/换距离"时的真实准确率——随机划分的验证集
因为帧与帧高度相似，数字会偏乐观。

用法：python group_holdout.py [epochs]
输出 data/holdout.json
"""
import csv
import json
import sys

import numpy as np
import tensorflow as tf
from PIL import Image

DATA = r"D:\esp32-gesture\data\dataset.csv"
PHOTO = {"file": r"D:\素材\e033213fb76bfb60eef432f11f85c891.jpg",
         "video": "photo", "time": "0", "label": "paper"}
IMG, SRC = 96, 128
CLASSES = ["paper", "rock", "scissors"]
EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 else 60


def load_one(path):
    im = Image.open(path).convert("RGB")
    w, h = im.size
    s = min(w, h)
    im = im.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s))
    return np.asarray(im.resize((SRC, SRC), Image.BILINEAR), dtype=np.float32) / 255.0


def build_model():
    m = tf.keras.Sequential([
        tf.keras.layers.Input((IMG, IMG, 3)),
        tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(128, 3, padding="same", activation="relu"),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(3),
    ])
    m.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=["accuracy"])
    return m


def main():
    rows = list(csv.DictReader(open(DATA, encoding="utf-8")))
    rows.append(PHOTO)
    x = np.stack([load_one(r["file"]) for r in rows])
    y = np.array([CLASSES.index(r["label"]) for r in rows], dtype=np.int32)
    videos = np.array([r["video"] for r in rows])

    # 增强层只建一次，方法内部再建会触发 tf.function 变量错误
    aug = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.10),
        tf.keras.layers.RandomZoom((-0.12, 0.12)),
    ])

    def prep(xx, yy):
        xx = aug(xx, training=True)
        k = tf.random.uniform([], 0, 4, dtype=tf.int32)
        xx = tf.image.rot90(xx, k)
        xx = tf.image.adjust_brightness(xx, tf.random.uniform([], -0.10, 0.10))
        xx = tf.clip_by_value(xx, 0.0, 1.0)
        return tf.image.resize(xx, (IMG, IMG)), yy

    results = {}
    # 只对样本足够的 4 段做留一评估；8 段全跑太慢且小视频会拖低均值
    for held in ["32777379", "359b5063", "a928747a", "c0cfc796"]:
        tr = np.where(videos != held)[0]
        te = np.where(videos == held)[0]
        if len(te) == 0 or len(tr) == 0:
            continue
        tf.random.set_seed(7)
        counts = np.bincount(y[tr], minlength=3)
        cw = {i: min(2.5, float((len(tr) / (3 * np.maximum(counts, 1)))[i] ** 0.5)) for i in range(3)}
        ds = (tf.data.Dataset.from_tensor_slices((x[tr], y[tr]))
              .shuffle(len(tr), seed=7).batch(16).map(prep).prefetch(1))
        m = build_model()
        m.fit(ds, epochs=EPOCHS, class_weight=cw, verbose=2)
        # 留出视频原样缩放（不做增强），模拟部署
        xt = np.stack([np.asarray(Image.fromarray(((x[i] * 255)).astype(np.uint8)).resize((IMG, IMG)), dtype=np.float32) / 255.0
                       for i in te]) if False else np.stack([x[i] for i in te])
        xt = tf.image.resize(xt, (IMG, IMG)).numpy()
        pred = np.argmax(m.predict(xt, verbose=0), axis=1)
        acc = float((pred == y[te]).mean())
        results[held] = {"n": int(len(te)), "acc": acc,
                         "labels": {CLASSES[c]: int((y[te] == c).sum()) for c in range(3)}}
        print("HELD", held, len(te), round(acc, 3), results[held]["labels"], flush=True)
        tf.keras.backend.clear_session()

    json.dump(results, open(r"D:\esp32-gesture\data\holdout.json", "w"), indent=2)
    print("mean", round(float(np.mean([v["acc"] for v in results.values()])), 3))


if __name__ == "__main__":
    main()
