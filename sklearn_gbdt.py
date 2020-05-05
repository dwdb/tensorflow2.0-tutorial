import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

dataset = load_iris()
features = dataset.feature_names

X_train, X_test, y_train, y_test = train_test_split(
    dataset.data, dataset.target, train_size=0.8, test_size=0.2, random_state=188)

gbc = GradientBoostingClassifier(
    loss='deviance',
    learning_rate=0.1,
    n_estimators=200,
    max_depth=4,
    min_samples_split=2,
    verbose=1)
gbc.fit(X_train, y_train)

y_pred = gbc.predict(X_test)
report = classification_report(y_test, gbc.predict(X_test), output_dict=False)
print(report)

sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, robust=True)
plt.show()
