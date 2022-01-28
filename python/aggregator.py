import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_federated as tff
import numpy as np
import os
from  functools import reduce
from typing import List, Tuple

import client

EPOCHS = 20
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


def bytes2model(wb_all: bytes) -> List[np.ndarray]:
    fw = np.frombuffer(wb_all[:31360], dtype=np.dtype(
        'float32'), count=-1, offset=0).reshape(784, 10)
    lw = np.frombuffer(wb_all[-40:], dtype=np.dtype(
        'float32'), count=-1, offset=0).reshape(10)
    ret = [fw, lw]
    return ret


def bmodels2weights(bmodels: List[bytes]) -> List[List[np.ndarray]]:
    models = []
    for bmodel in bmodels:
        model = bytes2model(bmodel)
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


def create_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        #tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.Dense(10, kernel_initializer='zeros'),
        tf.keras.layers.Softmax()
    ])
    model.compile(
        #optimizer=tf.keras.optimizers.Adam(0.001),
        optimizer=tf.keras.optimizers.SGD(learning_rate=1.0),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    )

    return model


# API
def aggregate(nodes_num: int, bmodels: List[bytes], nums: List[int]) -> Tuple[List[np.ndarray], str, str]:
    client_models = bmodels2weights(bmodels)

    fws = []
    lws = []
    for (weights, n) in zip(client_models, nums):
        fw = weights[0]
        lw = weights[1]
        fws.append(fw * n)
        lws.append(lw * n)

    fw_average = sum(fws) / float(sum(nums))
    lw_average = sum(lws) / float(sum(nums))


    aggregated = [fw_average, lw_average]

    """
    loss = 0
    acc = 0
    datanum = 0
    for i, test_data in enumerate(client.federated_test_data):
        if i == leader_id:
            continue

        datasize = len(list(test_data))
        l, a = model.evaluate(test_data, verbose=2)
        loss += l * datasize
        acc += a * datasize
        datanum += datasize
    
    loss = loss / float(datanum)
    acc = acc /float(datanum)
    """

    model = create_model()
    model.layers[1].set_weights(aggregated)
    dataset_sum = reduce(lambda x,y: x.concatenate(y), client.federated_test_data)

    """
    if leader_id == 0:
        dataset_sum = reduce(lambda x,y: x.concatenate(y), federated_test_data[leader_id+1:])
    else:
        dataset_sum = reduce(lambda x,y: x.concatenate(y), federated_test_data[:leader_id])
        dataset_sum = reduce(lambda x,y: x.concatenate(y), federated_test_data[leader_id+1:], dataset_sum)
    """

    loss, acc = model.evaluate(dataset_sum, verbose=2)

    loss_str = str(loss)
    acc_str = "{:.4f}".format(100*acc)


    return (aggregated, loss_str, acc_str)


def initial_learn() -> List[np.ndarray]:
    ret_weights = [np.random.normal(loc=0.0, scale=0.1, size=(784, 10)).astype(np.float32), np.random.normal(loc=0.0, scale=0.1, size=(10)).astype(np.float32)] 
    # ret_weights = [np.zeros((784, 10), dtype=np.float32), np.zeros((10), dtype=np.float32)]
    return ret_weights
