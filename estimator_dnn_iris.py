import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.python import tf_logging

tf_logging.set_verbosity(tf_logging.INFO)
from_tensor_slices = tf.data.Dataset.from_tensor_slices

columns = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'Species']
species = ['Setosa', 'Versicolor', 'Virginica']
train_path = keras.utils.get_file(
    'iris_training.csv',
    'https://storage.googleapis.com/download.tensorflow.org/data/iris_training.csv')
test_path = keras.utils.get_file(
    'iris_test.csv',
    'https://storage.googleapis.com/download.tensorflow.org/data/iris_test.csv')
train_x = pd.read_csv(train_path, names=columns, header=0)
test_x = pd.read_csv(test_path, names=columns, header=0)
train_y = train_x.pop('Species')
test_y = test_x.pop('Species')

feature_columns = [tf.feature_column.numeric_column(k) for k in train_x]
model = tf.estimator.DNNClassifier(
    hidden_units=[30, 10], feature_columns=feature_columns, n_classes=3)

input_fn = lambda: from_tensor_slices((dict(train_x), train_y)).shuffle(
    10000).repeat().batch(256)
model.train(input_fn=input_fn, steps=5000)

# def input_fn(features, labels=None, training=True, batch_size=256):
#     """输入函数"""
#     if labels is None:
#         dataset = tf.data.Dataset.from_tensor_slices(dict(features))
#     else:
#         dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))
#     if training:
#         dataset = dataset.shuffle(1000).repeat()
#     return dataset.batch(batch_size)


feature_columns = [tf.feature_column.numeric_column(k) for k in train]

classifier = tf.estimator.DNNClassifier(
    feature_columns=feature_columns,
    hidden_units=[30, 10],
    n_classes=3)
classifier.train(make_input_fn(train, train_y, training=True), steps=5000)

eval_result = classifier.evaluate(make_input_fn(test, test_y, training=False))
print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))

# # 由模型生成预测
expected = ['Setosa', 'Versicolor', 'Virginica']
predict_x = {
    'SepalLength': [5.1, 5.9, 6.9],
    'SepalWidth': [3.3, 3.0, 3.1],
    'PetalLength': [1.7, 4.2, 5.4],
    'PetalWidth': [0.5, 1.5, 2.1],
}

input_fn = lambda: from_tensor_slices((dict(predict_x))).batch(256)
predictions = model.predict(input_fn)
for pred_dict, expec in zip(predictions, expected):
    class_id = pred_dict['class_ids'][0]
    probability = pred_dict['probabilities'][class_id]
    print('Prediction is "{}" ({:.1f}%), expected "{}"'.format(
        species[class_id], 100 * probability, expec))
