import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from sklearn.metrics import roc_curve
from tensorflow import keras

tf.random.set_seed(123)

path_train = keras.utils.get_file(
    'titanic_train.csv',
    'https://storage.googleapis.com/tf-datasets/titanic/train.csv')
path_eval = keras.utils.get_file(
    'titanic_test.csv',
    'https://storage.googleapis.com/tf-datasets/titanic/eval.csv')
x_train, x_eval = pd.read_csv(path_train), pd.read_csv(path_eval)
y_train, y_eval = x_train.pop('survived'), x_eval.pop('survived')


def get_feature_columns(categories, numerics, dataset):
    feature_columns = []
    for feature in categories:
        vocab = dataset[feature].unique()
        feature = tf.feature_column.indicator_column(
            tf.feature_column.categorical_column_with_vocabulary_list(feature, vocab))
        feature_columns.append(feature)
    for feature in numerics:
        feature_columns.append(tf.feature_column.numeric_column(
            feature, dtype=tf.float32))
    return feature_columns


def make_input_fn(X, y, n_epochs=None, shuffle=True):
    def input_fn():
        dataset = tf.data.Dataset.from_tensor_slices((dict(X), y))
        if shuffle:
            dataset = dataset.shuffle(len(X))
        dataset = dataset.repeat(n_epochs).batch(len(X))
        return dataset

    return input_fn


CATEGORICAL_COLUMNS = ['sex', 'n_siblings_spouses', 'parch', 'class', 'deck',
                       'embark_town', 'alone']
NUMERIC_COLUMNS = ['age', 'fare']
feature_columns = get_feature_columns(CATEGORICAL_COLUMNS, NUMERIC_COLUMNS, x_train)
train_input_fn = make_input_fn(x_train, y_train)
eval_input_fn = make_input_fn(x_eval, y_eval, shuffle=False, n_epochs=1)

est = tf.estimator.LinearClassifier(feature_columns)
est.train(train_input_fn, max_steps=100)
result = est.evaluate(eval_input_fn)
print(pd.Series(result))

est = tf.estimator.BoostedTreesClassifier(feature_columns, n_batches_per_layer=1)
est.train(train_input_fn, max_steps=100)
result = est.evaluate(eval_input_fn)
print(pd.Series(result))

pred_dicts = list(est.predict(eval_input_fn))
probs = pd.Series([pred['probabilities'][1] for pred in pred_dicts])

probs.plot(kind='hist', bins=20, title='predicted probabilities')
plt.show()
fpr, tpr, _ = roc_curve(y_eval, probs)
plt.plot(fpr, tpr)
plt.title('ROC curve')
plt.xlabel('false positive rate')
plt.ylabel('true positive rate')
plt.xlim(0, )
plt.ylim(0, )
plt.show()
