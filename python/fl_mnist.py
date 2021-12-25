import collections
import socket
import requests
from requests.exceptions import Timeout
from http.client import RemoteDisconnected
from urllib3.util import Retry
from requests.adapters import HTTPAdapter

import json
import base64
import sys

import numpy as np
import tensorflow as tf
import tensorflow_federated as tff
from matplotlib import pyplot as plt

NUM_CLIENTS = 10
NUM_EPOCHS = 5
BATCH_SIZE = 20
SHUFFLE_BUFFER = 100
PREFETCH_BUFFER = 10
M_SIZE = 3136
LM_SIZE = 40


def preprocess(dataset):

    def batch_format_fn(element):
        """Flatten a batch `pixels` and return the features as an `OrderedDict`."""
        return collections.OrderedDict(
            x=tf.reshape(element['pixels'], [-1, 784]),
            y=tf.reshape(element['label'], [-1, 1]))

    return dataset.repeat(NUM_EPOCHS).shuffle(SHUFFLE_BUFFER, seed=1).batch(
        BATCH_SIZE).map(batch_format_fn).prefetch(PREFETCH_BUFFER)


emnist_train, emnist_test = tff.simulation.datasets.emnist.load_data()
sample_clients = emnist_train.client_ids[0:NUM_CLIENTS]

example_dataset = emnist_train.create_tf_dataset_for_client(
    emnist_train.client_ids[0])
print(example_dataset)
preprocessed_example_dataset = preprocess(example_dataset)


def make_federated_data(client_data, client_ids) -> [tf.data.Dataset]:
    return [
        preprocess(client_data.create_tf_dataset_for_client(x))
        for x in client_ids
    ]


def create_keras_model():
    return tf.keras.models.Sequential([
        tf.keras.layers.InputLayer(input_shape=(784,)),
        tf.keras.layers.Dense(10, kernel_initializer='zeros'),
        tf.keras.layers.Softmax()
    ])


def model_fn():
    # We _must_ create a new model here, and _not_ capture it from an external
    # scope. TFF will call this within different graph contexts.
    keras_model = create_keras_model()
    return tff.learning.from_keras_model(
        keras_model,
        input_spec=preprocessed_example_dataset.element_spec,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])


def main():
    federated_train_data = make_federated_data(emnist_train, sample_clients)

    print('Number of client datasets: {l}'.format(l=len(federated_train_data)))
    print('First dataset: {d}'.format(d=federated_train_data[0]))

    iterative_process = tff.learning.build_federated_averaging_process(
        model_fn,
        client_optimizer_fn=lambda: tf.keras.optimizers.SGD(
            learning_rate=0.02),
        server_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=1.0))

    # @test {"skip": true}
    logdir = "/tmp/logs/scalars/training/"
    summary_writer = tf.summary.create_file_writer(logdir)
    # 中央のサーバ状態を作成
    state = iterative_process.initialize()

    # 10 Roundで学習する
    NUM_ROUNDS = 20
    for round_num in range(NUM_ROUNDS):
        state, metrics = iterative_process.next(state, federated_train_data)
        # (784, 10) -> (10, 784)
        weights = state.model.trainable[0]

        # バイナリ化した重み
        bweights = weights.tobytes()
        # 4bytes  4 * 784 * 10= 31360
        # print("bweights: {}".format(len(bweights)))
        # 最後だけ, (10, 1) -> (1, 10)
        last_weight = state.model.trainable[1]
        # print("weight:  {}".format(weights[0].shape))
        lastb_weight = last_weight.tobytes()

        weights_all = bweights + lastb_weight

        session = requests.Session()
        retries = Retry(total=5,  # リトライ回数
                        backoff_factor=1,  # sleep時間
                        status_forcelist=[500, 502, 503, 504])
        session.mount("http://", HTTPAdapter(max_retries=retries))

        headers = {'Content-type': "application/json"}
        payload = {'model': base64.b64encode(weights_all).decode('utf-8')}
        try:
            r = session.post('http://localhost:8888/upload',
                             data=json.dumps(payload), headers=headers, stream=True, timeout=(10.0, 30.0))
            print(r)
        except requests.exceptions.ConnectionError:
            sys.exit()

        print('round {:2d}, metrics={}'.format(round_num + 1, metrics))
    # バイナリ化した重みを復元 -> len10のndarray
    # np.frombuffer(weights, dtype=np.dtype('float32'), count=-1, offset=0)

    # @test {"skip": true}
# with summary_writer.as_default():
#    for round_num in range(1, NUM_ROUNDS):
#        state, metrics = iterative_process.next(state, federated_train_data)
#        for name, value in metrics['train'].items():
#            tf.summary.scalar(name, value, step=round_num)


if __name__ == '__main__':
    main()
