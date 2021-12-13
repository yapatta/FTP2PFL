import collections
import functools
from absl import app
from absl import flags
import numpy as np
import tensorflow as tf
import tensorflow_federated as tff

from cleverhans.utils_keras import KerasModelWrapper

import random
import sys
import simple_fedavg_tf
import simple_fedavg_tff
import os

np.set_printoptions(precision=None, suppress=None)

# Training hyperparameters
flags.DEFINE_integer('total_rounds', 50, 'Number of total training rounds.')
flags.DEFINE_integer('rounds_per_eval', 1, 'How often to evaluate')
flags.DEFINE_integer('train_clients_per_round', 1,
                     'How many clients to sample per round.')
flags.DEFINE_integer('client_epochs_per_round', 1,
                     'Number of epochs in the client to take per round.')
flags.DEFINE_integer('batch_size', 16, 'Batch size used on the client.')
flags.DEFINE_integer('test_batch_size', 128, 'Minibatch size of test data.')

# Optimizer configuration (this defines one or more flags per optimizer).
flags.DEFINE_float('server_learning_rate', 0.0005, 'Server learning rate.')
flags.DEFINE_float('client_learning_rate', 0.0005, 'Client learning rate.')

FLAGS = flags.FLAGS


def create_vgg19_model():
    model = tf.keras.applications.VGG19(include_top=True,
                                        weights=None,
                                        input_shape=(32, 32, 3),
                                        classes=100)
    return model


def get_cifar100_dataset():
    cifar100_train, cifar100_test = tff.simulation.datasets.cifar100.load_data()

    def element_fn(element):
        return collections.OrderedDict(
            x=tf.expand_dims(element['image'], -1), y=element['label'])

    def preprocess_train_dataset(dataset):
        # Use buffer_size same as the maximum client dataset size,
        # 418 for Federated EMNIST
        return dataset.map(element_fn).shuffle(buffer_size=418).repeat(
            count=FLAGS.client_epochs_per_round)  # .batch(
        # FLAGS.batch_size, drop_remainder=False)

    def preprocess_test_dataset(dataset):
        return dataset.map(element_fn).batch(
            FLAGS.test_batch_size, drop_remainder=False)

    cifar100_train = cifar100_train.preprocess(preprocess_train_dataset)
    cifar100_test = preprocess_test_dataset(
        cifar100_test.create_tf_dataset_from_all_clients())
    return cifar100_train, cifar100_test


def server_optimizer_fn():
    return tf.keras.optimizers.SGD(learning_rate=FLAGS.server_learning_rate)


def client_optimizer_fn():
    return tf.keras.optimizers.Adam(learning_rate=FLAGS.client_learning_rate)


def main(argv):
    if len(argv) > 1:
        raise app.UsageError('Too many command-line arguments.')
    client_devices = tf.config.list_logical_devices('GPU')
    print(client_devices)
    server_device = tf.config.list_logical_devices('CPU')[0]
    tff.backends.native.set_local_python_execution_context(
        server_tf_device=server_device, client_tf_devices=client_devices)

    train_data, test_data = get_cifar100_dataset()

    def tff_model_fn():
        """Constructs a fully initialized model for use in federated averaging."""
        # keras_model = create_original_fedavg_cnn_model(only_digits=False)
        keras_model = create_vgg19_model()
        # keras_model.summary()
        loss = tf.keras.losses.SparseCategoricalCrossentropy()
        return simple_fedavg_tf.KerasModelWrapper(keras_model,
                                                  test_data.element_spec, loss)

    iterative_process = simple_fedavg_tff.build_federated_averaging_process(
        tff_model_fn, server_optimizer_fn, client_optimizer_fn)
    server_state = iterative_process.initialize()

    metric = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')
    model = tff_model_fn()

    for round_num in range(FLAGS.total_rounds):
        sampled_clients = np.random.choice(
            train_data.client_ids,  size=FLAGS.train_clients_per_round,   replace=False)
        sampled_train_data = [train_data.create_tf_dataset_for_client(client).batch(FLAGS.batch_size, drop_remainder=False)
                              for client in sampled_clients
                              ]

        server_state, train_metrics = iterative_process.next(
            server_state, sampled_train_data)

        print(f'Round {round_num} training loss: {train_metrics}')
        if round_num % FLAGS.rounds_per_eval == 0:
            model.from_weights(server_state.model_weights)
            accuracy = simple_fedavg_tf.keras_evaluate(model.keras_model, test_data,
                                                       metric)


if __name__ == '__main__':
    app.run(main)
