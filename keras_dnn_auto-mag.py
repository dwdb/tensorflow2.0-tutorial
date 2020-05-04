import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tensorflow import keras

dataset_path = keras.utils.get_file(
    'auto-mpg.data',
    'http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data')

column_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight',
                'Acceleration', 'Model Year', 'Origin']
dataset = pd.read_csv(dataset_path, ' ', header=None, names=column_names, na_values='?',
                      comment='\t', skipinitialspace=True)
dataset = dataset.dropna()
# 类别转为onehot编码
origin = dataset.pop('Origin')
dataset['USA'] = (origin == 1) * 1.0
dataset['Europe'] = (origin == 2) * 1.0
dataset['Japan'] = (origin == 3) * 1.0

# 分割数据集
train_dataset = dataset.sample(frac=0.8, random_state=0)
test_dataset = dataset.drop(train_dataset.index)

# 查看列之间的分布
sns.pairplot(train_dataset[["MPG", "Cylinders", "Displacement", "Weight"]],
             diag_kind="kde")
# plt.show()
# 总体数据统计-均值、方差、四分位、最大最小等
train_stats = train_dataset.describe()
train_stats.pop("MPG")
train_stats = train_stats.transpose()
print(train_stats)

train_labels = train_dataset.pop('MPG')
test_labels = test_dataset.pop('MPG')

# 数据标准化
train_dataset = (train_dataset - train_stats['mean']) / train_stats['std']
test_dataset = (test_dataset - train_stats['mean']) / train_stats['std']

model = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=[len(train_dataset.keys())]),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(1)])
model.summary()

model.compile(loss='mse',
              optimizer=keras.optimizers.RMSprop(lr=0.001),
              metrics=['mae', 'mse'])


class PrintDot(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if epoch % 100 == 0:
            print('')
        print('.', end='')


history = model.fit(train_dataset, train_labels, epochs=1000, validation_split=0.2,
                    verbose=0, callbacks=[PrintDot()])
hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch

# mse、mae的key名称不一样，以下是tensorflow==1.14版本的key
print(hist.columns)

# MAE Curve
plt.figure()
plt.xlabel('Epoch')
plt.ylabel('Mean Abs Error [MPG]')
plt.plot(hist['epoch'], hist['mean_absolute_error'], label='Train Error')
plt.plot(hist['epoch'], hist['val_mean_absolute_error'], label='Val Error')
plt.ylim([0, 5])
plt.legend()

# MSE Curve
plt.figure()
plt.xlabel('Epoch')
plt.ylabel('Mean Square Error [$MPG^2$]')
plt.plot(hist['epoch'], hist['mean_squared_error'], label='Train Error')
plt.plot(hist['epoch'], hist['val_mean_squared_error'], label='Val Error')
plt.ylim([0, 20])
plt.legend()
plt.show()

loss, mae, mse = model.evaluate(test_dataset, test_labels, verbose=2)
print("Testing set Mean Abs Error: {:5.2f} MPG".format(mae))

test_predictions = model.predict(test_dataset).flatten()

plt.figure()
plt.scatter(test_labels, test_predictions)
plt.xlabel('True Values [MPG]')
plt.ylabel('Predictions [MPG]')
plt.axis('equal')
plt.axis('square')
plt.xlim([0, plt.xlim()[1]])
plt.ylim([0, plt.ylim()[1]])
plt.plot([-100, 100], [-100, 100])

plt.figure()
plt.hist(test_predictions - test_labels, bins=25)
plt.xlabel("Prediction Error [MPG]")
plt.ylabel("Count")
plt.show()
