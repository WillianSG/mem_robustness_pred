# -*- coding: utf-8 -*-
"""
@author: w.soares.girao@rug.nl
@university: University of Groningen
@group: Bio-Inspired Circuits and System

References:
[1] https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_diabetes.html
[2] https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
"""

from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
import numpy as np

X, y = datasets.load_diabetes(return_X_y = True)	# diabetes dataset

print('data points: {}, features: {}'.format(X.shape[0], X.shape[1]))
print('ex: {}, label: {}\n'.format(X[0], y[0]))

# spliting data into training and testing sets.

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 42)

print('training datapoints: {}'.format(len(X_train)))
print('test datapoints: {}'.format(len(X_test)))

regr = svm.SVR()

regr.fit(X_train, y_train)

for i in range(0, len(X_test)):
	pred = regr.predict([X_test[i]])
	print('prediction: {}, real: {}'.format(
		np.round(pred[0], 2),
		np.round(y_test[i], 2)))