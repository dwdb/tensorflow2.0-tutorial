import matplotlib.pyplot as plt
from tensorflow.python import keras
from tensorflow.python.keras.layers import Dense, Dropout, Embedding, CuDNNLSTM
from tensorflow.python.keras.preprocessing.sequence import pad_sequences

vocab = keras.datasets.imdb.get_word_index()
vocab['<PAD>'] = 0
vocab['<START>'] = 1
vocab['<UNK>'] = 2  # unknown
vocab['<UNUSED>'] = 3
reverse_vocab = {v: k for k, v in vocab.items()}


def decode_review(words):
    return ' '.join(reverse_vocab.get(w, '?') for w in words)


(x_train, y_train), (x_test, y_test) = keras.datasets.imdb.load_data()
x_train = pad_sequences(x_train, 100, padding='post', truncating='post',
                        value=vocab['<PAD>'])
x_test = pad_sequences(x_test, 100, padding='post', truncating='post',
                       value=vocab['<PAD>'])

# 使用预训练词向量
model = keras.Sequential([
    Embedding(len(vocab), 128),
    Dropout(0.720),
    CuDNNLSTM(200),
    Dropout(0.708),
    Dense(1, activation='sigmoid')
])
model.summary()

model.compile(optimizer=keras.optimizers.Adam(0.01),
              loss=keras.losses.binary_crossentropy,
              metrics=['accuracy'])

history = model.fit(x_train, y_train, epochs=5, batch_size=128,
                    validation_data=(x_test[:1000], y_test[:1000]),
                    verbose=1)

score = model.evaluate(x_test, y_test, verbose=2)
print('Test loss:', score[0])
print('Test accuracy: %g %%' % (score[1] * 100.))

x_axis = range(1, len(history.history['loss']) + 1)

plt.plot(x_axis, history.history['loss'], 'bo', label='Traning Loss')
plt.plot(x_axis, history.history['val_loss'], 'b', label='Validation Loss')

plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
