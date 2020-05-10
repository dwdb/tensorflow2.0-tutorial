from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import pandas as pd

df = pd.read_csv('hot_words_dataset.csv', encoding='gbk')
df.pop('tag')

df_train = df[~df['label'].isna()]
df_test = df[df['label'].isna()]

word_train = df_train.pop('word')
y_train = df_train.pop('label')
X_train = df_train.values

word_test = df_test.pop('word')
df_test.pop('label')
X_test = df_test.values

xgb = XGBClassifier()
xgb.fit(X_train, y_train)

print(accuracy_score(y_train, xgb.predict(X_train)))

for word, pred_label in zip(word_test, xgb.predict(X_test)):
    print(word, pred_label)
