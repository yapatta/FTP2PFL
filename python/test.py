import tensorflow as tf
import tensorflow_federated as tff

NUM_CLIENTS = 6
NUM_EPOCHS = 20
BATCH_SIZE = 128
SHUFFLE_BUFFER = 100
PREFETCH_BUFFER = 10

emnist_train, emnist_test = tff.simulation.datasets.emnist.load_data()
# <ParallelMapDataset  shapes: , types: >
sample_clients = emnist_train.client_ids[0:NUM_CLIENTS]

def preprocess(dataset):
    # OrderDict([('label', <tf.Tensor>), ('pixels', <tf.Tensor>)]) -> Tuple()
    def batch_format_fn(element):
        # <tf.Tensor: shape=(28,28), dtype=float32, numpy=28x28array>
        # element["pixels"]
        # <tf.Tensor: shape=(), dtype=int32, numpy=9>
        # element["label"]
        """Flatten a batch `pixels` and return the features as an `Tuple`."""
        return (tf.reshape(element['pixels'], [28, 28, 1]), tf.cast(element['label'], tf.uint8))

    return dataset.repeat(NUM_EPOCHS).shuffle(SHUFFLE_BUFFER, seed=1).batch(
        BATCH_SIZE).map(batch_format_fn).prefetch(PREFETCH_BUFFER)


def make_federated_data(client_data, client_ids) -> List[tf.data.Dataset]:
    return [
        preprocess(client_data.create_tf_dataset_for_client(x))
        for x in client_ids
    ]

