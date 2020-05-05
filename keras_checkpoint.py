import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.datasets import mnist
from tensorflow.python import keras

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_labels = train_labels[:1000]
test_labels = test_labels[:1000]
train_images = train_images[:1000].reshape(-1, 784) / 255.0
test_images = test_images[:1000].reshape(-1, 784) / 255.0


def create_model():
    dnn = tf.keras.models.Sequential([
        keras.layers.Dense(512, activation='relu', input_shape=(784,)),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(10, activation='softmax')
    ])
    dnn.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
    dnn.summary()
    return dnn


# 使用新的回调训练模型
cp_path = "model/checkpoint_1/cp.ckpt"
cp_callback = ModelCheckpoint(cp_path, verbose=1, save_weights_only=True, period=5)
model = create_model()
model.fit(train_images, train_labels, epochs=10, verbose=2,
          validation_data=(test_images, test_labels),
          callbacks=[cp_callback])

model = create_model()
loss, acc = model.evaluate(test_images, test_labels, verbose=2)
print("Untrained model, accuracy: {:5.2f}%".format(100 * acc))

# 加载权重，重新评估模型
model.load_weights(cp_path)
loss, acc = model.evaluate(test_images, test_labels, verbose=2)
print("Restored model, accuracy: {:5.2f}%".format(100 * acc))

# 每10次迭代保存一次模型
cp_path = 'model/checkpoint_2/cp-{epoch:04d}.ckpt'
cp_callback = ModelCheckpoint(cp_path, verbose=1, save_weights_only=True, period=10)
model = create_model()
model.save_weights(cp_path.format(epoch=0))
model.fit(train_images, train_labels, epochs=50, verbose=2,
          validation_data=(test_images, test_labels),
          callbacks=[cp_callback])

# 将模型保存为HDF5文件
model = create_model()
model.fit(train_images, train_labels, epochs=5)
model.save('model/h5/my_model.h5')
# 重新创建完全相同的模型，包括其权重和优化程序
new_model = keras.models.load_model('model/h5/my_model.h5')

# 显示网络结构
new_model.summary()
loss, acc = new_model.evaluate(test_images, test_labels, verbose=2)
print("Restored model, accuracy: {:5.2f}%".format(100 * acc))
