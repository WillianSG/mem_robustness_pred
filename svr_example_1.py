# -*- coding: utf-8 -*-
"""
@author: w.soares.girao@rug.nl
@university: University of Groningen
@group: Bio-Inspired Circuits and System

References:
[1] https://scikit-learn.org/stable/modules/svm.html#regression
"""

from sklearn import svm

X = [[0, 0], [2, 2]]	# data points
y = [0.5, 2.5]			# target values	

regr = svm.SVR()

regr.fit(X, y)

pred = regr.predict([[1, 1]])

print('prediction: ', pred[0])