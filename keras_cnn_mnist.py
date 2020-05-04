from tensorflow.python import keras
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

imag_rows, imag_cols = 28, 28
num_classes = 10
batch_size = 128
epochs = 20
lr = 0.01
# channel last
image_shape = (imag_rows, imag_cols, 1)

# 默认存储在~/.keras/dataset/mnist.npz
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

x_train = x_train.reshape(-1, *image_shape).astype('float32') / 255.0
x_test = x_test.reshape(-1, *image_shape).astype('float32') / 255.0

# build graph
model = keras.Sequential()

model.add(Conv2D(32, kernel_size=(5, 5), activation='relu', input_shape=image_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, kernel_size=(5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(500, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

model.compile(optimizer=keras.optimizers.SGD(lr),
              loss=keras.losses.sparse_categorical_crossentropy,
              metrics=['accuracy'])

model.fit(x_train, y_train, batch_size, epochs, verbose=2,
          validation_data=(x_test, y_test))

score = model.evaluate(x_test, y_test)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
