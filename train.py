import os, json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

DATA_DIR = "/dataset/dataset_blood_group"
IMG_SIZE = (128, 128)
BATCH = 32
EPOCHS = 30
SEED = 42
AUTOTUNE = tf.data.AUTOTUNE

# ---- 1. List all files and labels ----
class_names = sorted(d for d in os.listdir(DATA_DIR)
                     if os.path.isdir(os.path.join(DATA_DIR, d)))
num_classes = len(class_names)
paths, labels = [], []
for i, c in enumerate(class_names):
    for f in os.listdir(os.path.join(DATA_DIR, c)):
        paths.append(os.path.join(DATA_DIR, c, f))
        labels.append(i)
paths, labels = np.array(paths), np.array(labels)
print("Classes:", class_names, "| total images:", len(paths))

# ---- 2. Stratified split: 70% train, 15% val, 15% test ----
tr_p, tmp_p, tr_y, tmp_y = train_test_split(
    paths, labels, test_size=0.30, stratify=labels, random_state=SEED)
va_p, te_p, va_y, te_y = train_test_split(
    tmp_p, tmp_y, test_size=0.50, stratify=tmp_y, random_state=SEED)

print("train per class:", np.bincount(tr_y))
print("val   per class:", np.bincount(va_y))
print("test  per class:", np.bincount(te_y))

# ---- 3. Data pipeline (BMP -> RGB -> resize -> grayscale) ----
def load(path, label):
    img = tf.io.read_file(path)
    img = tf.io.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.resize(img, IMG_SIZE)
    img = tf.image.rgb_to_grayscale(img)
    img.set_shape(IMG_SIZE + (1,))
    return img, label

def make_ds(p, y, training):
    ds = tf.data.Dataset.from_tensor_slices((p, y))
    ds = ds.map(load, num_parallel_calls=AUTOTUNE).cache()
    if training:
        ds = ds.shuffle(len(p), seed=SEED)
    return ds.batch(BATCH).prefetch(AUTOTUNE)

train_ds = make_ds(tr_p, tr_y, True)
val_ds = make_ds(va_p, va_y, False)
test_ds = make_ds(te_p, te_y, False)

# ---- 4. Model ----
model = models.Sequential([
    layers.Input(shape=IMG_SIZE + (1,)),
    layers.Rescaling(1.0 / 255),
    layers.RandomRotation(0.05),
    layers.RandomZoom(0.1),
    layers.Conv2D(32, 3, activation="relu", padding="same"),
    layers.BatchNormalization(),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, activation="relu", padding="same"),
    layers.BatchNormalization(),
    layers.MaxPooling2D(),
    layers.Conv2D(128, 3, activation="relu", padding="same"),
    layers.BatchNormalization(),
    layers.MaxPooling2D(),
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation="softmax"),
])
model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=6, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(patience=3, factor=0.5),
]
history = model.fit(train_ds, validation_data=val_ds,
                    epochs=EPOCHS, callbacks=callbacks)

# ---- 5. Evaluate on the test set ----
y_true, y_pred = [], []
for x, y in test_ds:
    probs = model.predict(x, verbose=0)
    y_pred.extend(np.argmax(probs, axis=1))
    y_true.extend(y.numpy())

print("\nTest accuracy:", np.mean(np.array(y_true) == np.array(y_pred)))
print(classification_report(y_true, y_pred, labels=list(range(num_classes)),
                            target_names=class_names, zero_division=0))

cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
plt.figure(figsize=(7, 6))
plt.imshow(cm, cmap="Blues")
plt.title("Confusion Matrix")
plt.xticks(range(num_classes), class_names, rotation=45)
plt.yticks(range(num_classes), class_names)
for i in range(num_classes):
    for j in range(num_classes):
        plt.text(j, i, cm[i, j], ha="center", va="center")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.colorbar()
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
plt.show()

plt.figure(figsize=(9, 3.5))
plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label="train")
plt.plot(history.history["val_accuracy"], label="val")
plt.title("Accuracy"); plt.legend()
plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label="train")
plt.plot(history.history["val_loss"], label="val")
plt.title("Loss"); plt.legend()
plt.tight_layout()
plt.savefig("training_curves.png", dpi=150)
plt.show()

model.save("blood_cnn.keras")
with open("class_names.json", "w") as f:
    json.dump(class_names, f)
print("Saved model")