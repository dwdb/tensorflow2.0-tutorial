import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.platform import tf_logging

tf_logging.set_verbosity(tf_logging.INFO)

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train.reshape(-1, 784).astype(np.float32) / 255.
x_test = x_test.reshape(-1, 784).astype(np.float32) / 255.
y_train = y_train.astype(np.int32)
y_test = y_test.astype(np.int32)


def make_input_fn(features, labels=None, training=True, batch_size=256):
    def input_fn():
        if labels is None:
            dataset = tf.data.Dataset.from_tensor_slices(dict(features))
        else:
            dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))
        if training:
            dataset = dataset.shuffle(2000).repeat()
        return dataset.batch(batch_size)

    return input_fn


feature_columns = [tf.feature_column.numeric_column('x', shape=[784])]
estimator = tf.estimator.DNNClassifier(
    feature_columns=feature_columns,
    hidden_units=[500, 400, 400],
    n_classes=10,
    optimizer=keras.optimizers.Adam(0.001),
    model_dir='model/estimator')

estimator.train(
    input_fn=make_input_fn({'x': x_train}, y_train, training=True, batch_size=128),
    steps=10000)

# test_input_fn = tf.estimator.inputs.numpy_input_fn(
#     x={'image': mnist.test.images},
#     y=mnist.test.labels.astype(np.int32),
#     num_epochs=1,
#     batch_size=128,
#     shuffle=False)
#
# accuracy_score = estimator.evaluate(input_fn=test_input_fn)['accuracy']
# print('\nTest accuracy: %g %%' % (accuracy_score * 100))
