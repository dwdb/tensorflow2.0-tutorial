import numpy as np
import tensorflow as tf

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=128,
          epochs=15,
          verbose=2,
          validation_data=(x_test, y_test))

model.evaluate(x_test, y_test, verbose=2)

model_prob = tf.keras.models.Sequential([model, tf.keras.layers.Softmax()])
predict_prob = model_prob.predict(x_test)
predict_label = np.argmax(predict_prob, 1)
