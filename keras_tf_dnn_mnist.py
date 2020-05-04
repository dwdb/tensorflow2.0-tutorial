import os

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist_path = os.path.expanduser('~') + '/.keras/datasets/minist'
mnist = input_data.read_data_sets(mnist_path, one_hot=True)

input_x = tf.placeholder(tf.float32, shape=[None, 784])
input_y = tf.placeholder(tf.float32, shape=[None, 10])

net = tf.keras.layers.Dense(500, activation='relu')(input_x)
y = tf.keras.layers.Dense(10, activation='softmax')(net)

loss = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(input_y, y))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

acc_value = tf.reduce_mean(tf.keras.metrics.categorical_accuracy(input_y, y))

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    for i in range(10000):
        xs, ys = mnist.train.next_batch(100)
        _, loss_value = sess.run([train_step, loss], feed_dict={input_x: xs, input_y: ys})
        if (i + 1) % 1000 == 0:
            accuracy_prob = acc_value.eval(feed_dict={input_x: mnist.test.images,
                                                      input_y: mnist.test.labels})
            print(i + 1, '%g' % loss_value, ', Test accuracy:', accuracy_prob)
