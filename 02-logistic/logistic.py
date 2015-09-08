__author__ = 'rain'

import statsmodels.api as sm
import pandas as pd
from statsmodels.tools.tools import categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix
import numpy
from sklearn.tree import DecisionTreeClassifier


def get_data():
    return pd.read_csv("./bank/bank-full.csv", header=0, sep=";")

data = get_data()

data.job = LabelEncoder().fit_transform(data.job)
data.marital = LabelEncoder().fit_transform(data.marital)
data.education = LabelEncoder().fit_transform(data.education)
data.default = LabelEncoder().fit_transform(data.default)
data.housing = LabelEncoder().fit_transform(data.housing)
data.loan = LabelEncoder().fit_transform(data.loan)
data.month = LabelEncoder().fit_transform(data.month)
data.contact = LabelEncoder().fit_transform(data.contact)
data.poutcome = LabelEncoder().fit_transform(data.poutcome)

X = data.iloc[:, :-1]
y = data.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

clf = LogisticRegression()
clf.fit(X_train, y_train)
score = clf.score(X_test, y_test)

print confusion_matrix(y_test, clf.predict(X_test))
# [[11807   203]
#  [ 1243   311]]
# it's too bad

# data
print "DECISION TREE CLASSIFIER"
data = get_data()

data.job = LabelEncoder().fit_transform(data.job)
data.marital = LabelEncoder().fit_transform(data.marital)
data.education = LabelEncoder().fit_transform(data.education)
data.default = LabelEncoder().fit_transform(data.default)
data.housing = LabelEncoder().fit_transform(data.housing)
data.loan = LabelEncoder().fit_transform(data.loan)
data.month = LabelEncoder().fit_transform(data.month)
data.contact = LabelEncoder().fit_transform(data.contact)
data.poutcome = LabelEncoder().fit_transform(data.poutcome)

X = data.iloc[:, :-1]
y = data.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

clf_tree = DecisionTreeClassifier()
clf_tree.fit(X_train, y_train)
print confusion_matrix(y_test, clf_tree.predict(X_test))
