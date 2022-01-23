import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
from typing import List, ByteString
import os

CLIANT_NUM = 6
EPOCHS = 20
MODEL_FILE = "parent.model"
CLIANT_MODEL_FILES = ["client{}.model".format(i) for i in range(CLIANT_NUM)]

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


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


def bytes2model(wb_all: ByteString) -> List[np.ndarray]:
    fw = np.frombuffer(wb_all[:31360], dtype=np.dtype(
        'float32'), count=-1, offset=0).reshape(784, 10)
    lw = np.frombuffer(wb_all[-40:], dtype=np.dtype(
        'float32'), count=-1, offset=0).reshape(10)
    ret = [fw, lw]
    return ret


def bytefile_to_model(fn: str) -> List[np.ndarray]:
    if not os.path.isfile(fn):
        return list()

    with open(fn, 'rb') as f:
        wb_all = f.read()
        return bytes2model(wb_all)


def bmodels2weights(bmodels: List[ByteString]) -> List[List[np.ndarray]]:
    models = []
    for bmodel in bmodels:
        model = bytes2model(bmodel)
        if len(model) == 0:
            continue
        models.append(model)

    return models


# クライアントのモデル, 存在する場合のみ増やす
def load_cliant_models() -> List[List[np.ndarray]]:
    models = []
    for mf in CLIANT_MODEL_FILES:
        model = bytefile_to_model(mf)
        if len(model) == 0:
            continue
        models.append(model)

    return models


def byte_weights(weights: List[np.ndarray]):
    # (784, 10)
    fwb = weights[0].tobytes()
    # (10, )
    lwb = weights[1].tobytes()

    wb_all = fwb + lwb

    return wb_all


def save_aggregated_weights(weights: List[np.ndarray]):
    # (784, 10)
    fwb = weights[0].tobytes()
    # (10, )
    lwb = weights[1].tobytes()

    wb_all = fwb + lwb

    with open(MODEL_FILE, 'wb') as f:
        f.write(wb_all)


def aggregate(bmodels: List[ByteString]) -> List[np.ndarray]:
    client_models = bmodels2weights(bmodels)

    fws = []
    lws = []

    # client_models = load_cliant_models()

    for weights in client_models:
        fw = weights[0]
        lw = weights[1]
        fws.append(fw)
        lws.append(lw)

    fw_average = sum(fws) / float(len(fws))
    lw_average = sum(lws) / float(len(lws))

    ret = [fw_average, lw_average]
    save_aggregated_weights(ret)

    return ret


def load_parent_model() -> List[np.ndarray]:
    model = bytefile_to_model(MODEL_FILE)
    return model


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


def initial_learn() -> List[np.ndarray]:
    ds_train, ds_test = fetch_train_test_data()
    model.fit(
        ds_train,
        epochs=EPOCHS,
        validation_data=ds_test,
    )

    # save_aggregated_weights(model.layers[1].get_weights())

    return model.layers[1].get_weights()
