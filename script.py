__author__ = 'rain'

from sklearn.datasets import load_boston
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
boston = load_boston()
data = boston['data']
X, y = data[:, :-1], data[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
print boston['DESCR']
clf_linear = LinearRegression()
clf_linear.fit(X_train, y_train)
linear_score = clf_linear.score(X_test, y_test)

clf_ridge = Ridge(alpha=1.0)
clf_ridge.fit(X_train, y_train)
ridge_score = clf_ridge.score(X_test, y_test)
print y_test
print clf_linear.predict(X_test)
print clf_ridge.predict(X_test)
print 1
