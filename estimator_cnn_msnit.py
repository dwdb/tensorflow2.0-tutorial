"""
tensorflow==1.14
"""

import os

import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

tf.logging.set_verbosity(tf.logging.INFO)


def lenet(x, is_training):
    x = tf.reshape(x, shape=[-1, 28, 28, 1])
    net = tf.layers.conv2d(x, 32, 5, activation=tf.nn.relu)
    net = tf.layers.max_pooling2d(net, 2, 2)
    net = tf.layers.conv2d(net, 64, 3, activation=tf.nn.relu)
    net = tf.layers.max_pooling2d(net, 2, 2)
    net = tf.contrib.layers.flatten(net)
    net = tf.layers.dense(net, 1014)
    net = tf.layers.dropout(net, rate=0.4, training=is_training)
    net = tf.layers.dense(net, 10)
    return net


def model_fn(features, labels, mode, params):
    logits = lenet(features['image'], mode == tf.estimator.ModeKeys.TRAIN)
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(
            mode=mode, predictions={'result': tf.argmax(logits, 1)})

    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits, labels=labels))
    train_op = tf.train.GradientDescentOptimizer(params['lr']).minimize(
        loss=loss, global_step=tf.train.get_global_step())

    eval_metric_ops = {'my_metric': tf.metrics.accuracy(tf.argmax(logits, 1), labels)}

    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, train_op=train_op, eval_metric_ops=eval_metric_ops)


mnist_path = os.path.expanduser('~') + '/.keras/datasets/minist'
mnist = input_data.read_data_sets(mnist_path, one_hot=False)

model_params = {'lr': 0.001}
estimator = tf.estimator.Estimator(model_fn=model_fn, params=model_params)

train_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={'image': mnist.train.images},
    y=mnist.train.labels.astype(np.int32),
    num_epochs=None,
    batch_size=128,
    shuffle=True
)

estimator.train(input_fn=train_input_fn, steps=30000)

test_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={'image': mnist.test.images},
    y=mnist.test.labels.astype(np.int32),
    num_epochs=1,
    batch_size=128,
    shuffle=False
)

test_results = estimator.evaluate(input_fn=test_input_fn)

accuracy_score = test_results['my_metric']
print('\nTest accuracy: %g %%' % (accuracy_score * 100))

predict_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={'image': mnist.test.images[:10]},
    num_epochs=1,
    shuffle=False)

predictions = estimator.predict(input_fn=predict_input_fn)

for i, p in enumerate(predictions):
    print('Predition %s: %s' % (i + 1, p['result']))
