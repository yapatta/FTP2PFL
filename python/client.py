import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_federated as tff
import numpy as np
import sys
import os
import random
from typing import List, Tuple

# for test
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

NUM_CLIENTS = 12
NUM_EPOCHS = 20
BATCH_SIZE = 20
SHUFFLE_BUFFER = 100
PREFETCH_BUFFER = 10

emnist_train, emnist_test = tff.simulation.datasets.emnist.load_data()
# <ParallelMapDataset  shapes: , types: >
sample_clients = emnist_train.client_ids[0:NUM_CLIENTS]


def preprocess(dataset):
    # OrderDict([('label', <tf.Tensor>), ('pixels', <tf.Tensor>)]) -> Tuple()
    def normalize_img(element):
        # <tf.Tensor: shape=(28,28), dtype=float32, numpy=28x28array>
        # element["pixels"]
        # <tf.Tensor: shape=(), dtype=int32, numpy=9>
        # element["label"]
        """Flatten a batch `pixels` and return the features as an `Tuple`."""
        return (tf.reshape(element['pixels'], [28, 28, 1]), tf.reshape(element['label'], [-1, 1]))
    return dataset.map(normalize_img).repeat(NUM_EPOCHS).shuffle(SHUFFLE_BUFFER, seed=1).batch(
        BATCH_SIZE).prefetch(PREFETCH_BUFFER)


def make_federated_data_for_id(client_data, id) -> tf.data.Dataset:
    return preprocess(client_data.create_tf_dataset_for_client(sample_clients[id]))


# fl dataset
def make_federated_data(client_data, client_ids) -> List[tf.data.Dataset]:
    return [
        preprocess(client_data.create_tf_dataset_for_client(x))
        for x in client_ids
    ]



def bytes2model(wb_all: bytes) -> List[np.ndarray]:
    fw = np.frombuffer(wb_all[:31360], dtype=np.dtype(
        'float32'), count=-1, offset=0).reshape(784, 10)
    lw = np.frombuffer(wb_all[-40:], dtype=np.dtype(
        'float32'), count=-1, offset=0).reshape(10)
    ret = [fw, lw]
    return ret



# normal dataset
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

def create_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        #tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.Dense(10, kernel_initializer='zeros'),
        tf.keras.layers.Softmax()
    ])
    model.compile(
        #optimizer=tf.keras.optimizers.Adam(0.001),
        optimizer=tf.keras.optimizers.SGD(learning_rate=0.02),
        #loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    )

    return model

def create_federated_test_data(l: int):
    return make_federated_data(emnist_test, sample_clients[:l])

federated_test_data = create_federated_test_data(NUM_CLIENTS)

def learn(id: int, bweights: bytes) -> Tuple[List[np.ndarray], int, str, str]:
    model = create_model()

    pw = bytes2model(bweights)
    model.layers[1].set_weights(pw)

    fl_data = make_federated_data_for_id(emnist_train, id)
    model.fit(
        fl_data,
        epochs=NUM_EPOCHS,
    )

    loss, acc = model.evaluate(make_federated_data_for_id(emnist_test, id), verbose=2)
    loss_str = str(loss)
    acc_str = "{:5.2f}".format(100*acc)

    return (model.layers[1].get_weights(), len(list(fl_data)), loss_str, acc_str)

