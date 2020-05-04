import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow import keras

# Load dataset.

train_data = keras.utils.get_file(
    'titanic_train.csv',
    'https://storage.googleapis.com/tf-datasets/titanic/train.csv'
)
eval_data = keras.utils.get_file(
    'titanic_test.csv',
    'https://storage.googleapis.com/tf-datasets/titanic/eval.csv'
)
dftrain = pd.read_csv(train_data)
dfeval = pd.read_csv(eval_data)

y_train = dftrain.pop('survived')
y_eval = dfeval.pop('survived')
print(dftrain.head())

dftrain.age.hist(bins=20)
plt.show()

dftrain.sex.value_counts().plot(kind='barh')
plt.show()

categories = 'sex n_siblings_spouses parch class deck embark_town alone'.split()
numerics = 'age fare'.split()
feature_columns = []
for feature in categories:
    feature = tf.feature_column.categorical_column_with_vocabulary_list(
        feature, dftrain[feature].unique())
    feature_columns.append(feature)
for feature in numerics:
    feature = tf.feature_column.numeric_column(feature, dtype=tf.float32)
    feature_columns.append(feature)
