"""Train rock/paper/scissors CNN -> int8 TFLite + model.h for ESP32-CAM.

Data: data/dataset.csv (470 frames from 8 videos) + one paper still.
Split: stratified 80/20. NOTE: frames from the same video are highly
correlated, so validation accuracy is optimistic. A real test needs new
recordings (different background / distance / hand).
"""
import csv
import json
import os

import numpy as np
import tensorflow as tf
from PIL import Image

DATA = r"D:\esp32-gesture\data"
MODEL_DIR = r"D:\esp32-gesture\model"
IMG = 96
SRC = 128
CLASSES = ["paper", "rock", "scissors"]
SEED = 7

os.makedirs(MODEL_DIR, exist_ok=True)


def square128(path):
    im = Image.open(path).convert("RGB")
    w, h = im.size
    s = min(w, h)
    im = im.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s))
    im = im.resize((SRC, SRC), Image.BILINEAR)
    return np.asarray(im, dtype=np.float32) / 255.0


def load():
    rows = list(csv.DictReader(open(os.path.join(DATA, "dataset.csv"), encoding="utf-8")))
    rows.append({"file": r"D:\素材\e033213fb76bfb60eef432f11f85c891.jpg",
                 "video": "photo", "time": "0", "label": "paper"})
    x = np.stack([square128(r["file"]) for r in rows])
    y = np.array([CLASSES.index(r["label"]) for r in rows], dtype=np.int32)
    return x, y


def split(x, y):
    rng = np.random.default_rng(SEED)
    tr, va = [], []
    for c in range(len(CLASSES)):
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        cut = max(1, int(len(idx) * 0.8))
        tr += list(idx[:cut])
        va += list(idx[cut:])
    return np.array(tr), np.array(va)


def main():
    tf.random.set_seed(SEED)
    x, y = load()
    tr, va = split(x, y)
    print("train", len(tr), "val", len(va), "counts", np.bincount(y))

    counts = np.bincount(y[tr], minlength=3)
    cw = {i: min(2.5, float((len(tr) / (3 * np.maximum(counts, 1)))[i] ** 0.5)) for i in range(3)}
    print("class_weight", cw)

    # NOTE: keras RandomBrightness/Contrast assume 0-255 inputs and destroy
    # [0,1] images, so photometric aug is done with tf.image on [0,1].
    aug = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.10),
        tf.keras.layers.RandomZoom((-0.12, 0.12)),
        tf.keras.layers.RandomTranslation(0.08, 0.08),
    ])

    def prep(xx, yy):
        xx = aug(xx, training=True)
        k = tf.random.uniform([], 0, 4, dtype=tf.int32)
        xx = tf.image.rot90(xx, k)   # 摄像头可能横装，补 90 度旋转不变性
        xx = tf.image.adjust_brightness(xx, tf.random.uniform([], -0.10, 0.10))
        xx = tf.image.adjust_contrast(xx, tf.random.uniform([], 0.8, 1.2))
        xx = tf.image.adjust_saturation(xx, tf.random.uniform([], 0.8, 1.2))
        xx = tf.clip_by_value(xx, 0.0, 1.0)
        return tf.image.resize(xx, (IMG, IMG)), yy

    ds = (tf.data.Dataset.from_tensor_slices((x[tr], y[tr]))
          .shuffle(len(tr), seed=SEED).batch(16).map(prep).prefetch(1))
    x_va = tf.image.resize(x[va], (IMG, IMG)).numpy()

    model = tf.keras.Sequential([
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
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=["accuracy"])
    model.fit(ds, epochs=300, validation_data=(x_va, y[va]), class_weight=cw, verbose=2,
              callbacks=[tf.keras.callbacks.EarlyStopping(patience=40, restore_best_weights=True),
                         tf.keras.callbacks.ReduceLROnPlateau(patience=15, factor=0.5)])
    print("float val", model.evaluate(x_va, y[va], verbose=0))

    def rep():
        for i in range(len(x[tr])):
            yield [tf.image.resize(x[tr][i:i + 1], (IMG, IMG))]

    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.representative_dataset = rep
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    conv.inference_input_type = tf.uint8
    conv.inference_output_type = tf.uint8
    blob = conv.convert()
    path = os.path.join(MODEL_DIR, "gesture_model.tflite")
    open(path, "wb").write(blob)
    print("tflite bytes", len(blob))

    interp = tf.lite.Interpreter(model_path=path)
    interp.allocate_tensors()
    inp, outp = interp.get_input_details()[0], interp.get_output_details()[0]
    conf = np.zeros((3, 3), dtype=int)
    for i in range(len(x_va)):
        q = (x_va[i] * 255).round().astype(np.uint8)[None, ...]
        interp.set_tensor(inp["index"], q)
        interp.invoke()
        pred = int(np.argmax(interp.get_tensor(outp["index"])[0]))
        conf[pred, y[va][i]] += 1
    print("int8 val acc", np.trace(conf) / len(x_va), f"({np.trace(conf)}/{len(x_va)})")
    print("confusion rows=pred cols=true", CLASSES)
    print(conf)

    open(os.path.join(MODEL_DIR, "labels.txt"), "w", encoding="utf-8").write("\n".join(CLASSES))
    json.dump(cw, open(os.path.join(MODEL_DIR, "class_weight.json"), "w"))

    # model.h for Arduino (PROGMEM array)
    lines = [",".join(str(b) for b in blob[i:i + 16]) for i in range(0, len(blob), 16)]
    hdr = ("// Auto-generated from gesture_model.tflite (%d bytes)\n"
           "// classes: paper, rock, scissors\n"
           "const unsigned char gesture_model[] PROGMEM = {\n%s\n};\n"
           "const unsigned int gesture_model_len = %d;\n" % (len(blob), ",\n".join(lines), len(blob)))
    open(os.path.join(MODEL_DIR, "model.h"), "w", encoding="utf-8").write(hdr)
    print("wrote model.h")


if __name__ == "__main__":
    main()
