import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import os
from typing import List

MODEL_FILE = "client{}.model"
PARENT_MODEL_FILE = "parent.model"
EPOCHS = 6


def fetch_train_test_data():
    (ds_train, ds_test), ds_info = tfds.load(
        'mnist',
        split=['train', 'test'],
        shuffle_files=True,
        as_supervised=True,
        with_info=True,
    )

    def normalize_img(image, label):
        """Normalizes images: `uint8` -> `float32`."""
        return tf.cast(image, tf.float32) / 255., label

    ds_train = ds_train.map(
        normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
    ds_train = ds_train.cache()
    ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
    ds_train = ds_train.batch(128)
    ds_train = ds_train.prefetch(tf.data.AUTOTUNE)

    ds_test = ds_test.map(
        normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
    ds_test = ds_test.batch(128)
    ds_test = ds_test.cache()
    ds_test = ds_test.prefetch(tf.data.AUTOTUNE)

    return ds_train, ds_test


model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Softmax()
    # tf.keras.layers.Dense(1, activation='softmax')
])
model.summary()
model.compile(
    optimizer=tf.keras.optimizers.Adam(0.001),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
)


def bytes_to_model(fn: str) -> List[np.ndarray]:
    if not os.path.isfile(fn):
        return list()

    with open(fn, 'rb') as f:
        wb_all = f.read()
        fw = np.frombuffer(wb_all[:31360], dtype=np.dtype(
            'float32'), count=-1, offset=0).reshape(784, 10)
        lw = np.frombuffer(wb_all[-40:], dtype=np.dtype(
            'float32'), count=-1, offset=0).reshape(10)
        ret = [fw, lw]
        return ret


def load_parent_model() -> List[np.ndarray]:
    return bytes_to_model(PARENT_MODEL_FILE)


def save_client_weights(weights: List[np.ndarray], id: int):
    # (784, 10)
    fwb = weights[0].tobytes()
    # (10, )
    lwb = weights[1].tobytes()

    wb_all = fwb + lwb

    with open(MODEL_FILE.format(id), 'wb') as f:
        f.write(wb_all)


def learn(id: int):
    pw = load_parent_model()
    if len(pw) == 0:
        os.exit(1)

    model.layers[1].set_weights(pw)

    ds_train, ds_test = fetch_train_test_data()
    model.fit(
        ds_train,
        epochs=EPOCHS,
        validation_data=ds_test,
    )

    save_client_weights(model.layers[1].get_weights(), id)
