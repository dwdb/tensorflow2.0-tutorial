import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.python import keras

gz = keras.utils.get_file('HIGGS.csv.gz',
                          'http://mlphysics.ics.uci.edu/data/higgs/HIGGS.csv.gz')
FEATURES = 28
dataset = tf.data.experimental.CsvDataset(gz, [0., ] * (FEATURES + 1), 'GZIP')


def pack_row(*row):
    return tf.stack(row[1:], 1), row[0]


packed_ds = dataset.batch(10000).map(pack_row).unbatch()
for features, label in packed_ds.batch(1000).take(1):
    print(features[0])
    plt.hist(features.numpy().flatten(), bins=101)
    plt.show()

N_VALIDATION = int(1e3)
N_TRAIN = int(1e4)
BUFFER_SIZE = int(1e4)
BATCH_SIZE = 500
STEPS_PER_EPOCH = N_TRAIN // BATCH_SIZE

validate_ds = packed_ds.take(N_VALIDATION).cache().batch(BATCH_SIZE)
train_ds = packed_ds.skip(N_VALIDATION).take(N_TRAIN).cache().shuffle(
    BUFFER_SIZE).repeat().batch(BATCH_SIZE)

# lr = lr  / (1 + epochs / 1000)
lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
    0.001,
    decay_steps=STEPS_PER_EPOCH * 1000,
    decay_rate=1,
    staircase=False)


def get_optimizer():
    return tf.keras.optimizers.Adam(lr_schedule)


def compile_and_fit(model, name, optimizer=None, max_epochs=10000):
    if optimizer is None:
        optimizer = keras.optimizers.Adam(lr_schedule)
    model.compile(
        optimizer=keras.optimizers.Adam(lr_schedule),
        loss=keras.losses.BinaryCrossentropy(from_logits=True),
        metric=[
            keras.losses.BinaryCrossentropy(True, name='binary_crossentropy'),
            'accuracy'
        ])

    model.summary()

    history = model.fit(
        train_ds,
        steps_per_epoch=STEPS_PER_EPOCH,
        epochs=10000,
        validation_data=validate_ds,
        verbose=0,
        callbacks=[
            keras.callbacks.EarlyStopping(monitor='val_binary_crossentropy',
                                          patience=200),
            keras.callbacks.TensorBoard('log')
        ])
    return history


model = keras.Sequential([
    keras.layers.Dense(16, activation='relu', input_shape=(FEATURES,)),
    keras.layers.Dense(1)])

size_histories = {}
