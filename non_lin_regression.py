import numpy as np
import matplotlib.pyplot as plt
import sklearn

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression

from numpy.polynomial.polynomial import polyfit
from sklearn.metrics import mean_squared_error, r2_score

# USEFUL LINKS
# https://scikit-learn.org/stable/auto_examples/linear_model/plot_ols.html

# ============================ pre-processing dataset ============================

X, Y = load_diabetes(return_X_y = True)

_feat_vars = {}

# calculating variance across features
for _kth in range(0, X.shape[1]):
	_ = [pow(sub[_kth], 2) for sub in X]
	_feat_vars[_kth] = np.std(_)

_1st = -1
_1key = 0
_2nd = -1
_2key = 0
_3rd = -1
_3key = 0

for key, val in _feat_vars.items():
	if val > _1st:
		_1st = val
		_1key = key
	elif val > _2nd:
		_2nd = val
		_2key = key
	elif val > _3rd:
		_3rd = val
		_3key = key
	else:
		pass

X_reduced = []

# using feature with top 3 biggest variances
for _kth in [_1key, _2key, _3key]:
	X_reduced.append([sub[_kth] for sub in X])

X_reduced = np.array(X_reduced).T

fig = plt.figure()

ax = fig.add_subplot(111, projection='3d')

x = [sub[0] for sub in X_reduced]
y = [sub[1] for sub in X_reduced]
z = [sub[2] for sub in X_reduced]

color_map = plt.get_cmap('spring')

Y = Y/np.max(Y)

scatter_plot = ax.scatter(x, y, z, c = Y, cmap = color_map)

plt.colorbar(scatter_plot, label = 'target value')

plt.title('top 3 highest var features')

plt.show()
plt.close()

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.05, random_state = 1)

# ============================ linear regression ============================

svr_lin = SVR(kernel = "linear", C = 100, gamma = "auto")

svr_lin.fit(X_train, Y_train)

y_predicted = []
y_expected = []

for i in range(0, len(X_test)):
	pred = svr_lin.predict([X_test[i]])

	y_predicted.append(np.round(pred[0], 3))
	y_expected.append(np.round(Y_test[i], 3))

errors = []

_mse = 0

for i in range(len(y_expected)):
	err = (y_expected[i] - y_predicted[i])**2
	errors.append(err)

	_mse += err

print('linear - MSE: ', np.round(_mse/len(y_expected), 3))
print("Mean squared error: %.2f" % mean_squared_error(y_expected, y_predicted))
# The coefficient of determination: 1 is perfect prediction
print("Coefficient of determination: %.2f\n" % r2_score(y_expected, y_predicted))


plt.title('linear SVR')
# plt.yticks(np.arange(0.0, 0.6, step = 0.05))
# plt.ylim(0.0, 0.5)

plt.yticks(np.arange(50, 400, step = 50))
plt.xticks(np.arange(50, 400, step = 50))

# plt.plot(errors)
# plt.xlabel('Predicted Value')
# plt.ylabel('Mean Squared Error')

b, m = polyfit(y_predicted, y_expected, 1)
plt.scatter(y_predicted, y_expected)
plt.plot(np.array(y_predicted), b + m * np.array(y_predicted), '-', color = 'k')

plt.show()

# ============================ non-linear regression 1 ============================

svr_poly = SVR(kernel = "poly", C = 100, gamma = "auto", degree = 3, epsilon = 0.1, coef0 = 1)

svr_poly.fit(X_train, Y_train)

y_predicted = []
y_expected = []

for i in range(0, len(X_test)):
	pred = svr_poly.predict([X_test[i]])

	y_predicted.append(np.round(pred[0], 3))
	y_expected.append(np.round(Y_test[i], 3))

errors = []

_mse = 0

for i in range(len(y_expected)):
	err = (y_expected[i] - y_predicted[i])**2
	errors.append(err)

	_mse += err

print('poly - MSE: ', np.round(_mse/len(y_expected), 3))
print("Mean squared error: %.2f" % mean_squared_error(y_expected, y_predicted))
# The coefficient of determination: 1 is perfect prediction
print("Coefficient of determination: %.2f\n" % r2_score(y_expected, y_predicted))

plt.title('poly SVR')
# plt.yticks(np.arange(0.0, 0.6, step = 0.05))
# plt.ylim(0.0, 0.5)

plt.yticks(np.arange(50, 400, step = 50))
plt.xticks(np.arange(50, 400, step = 50))

# plt.plot(errors)
# plt.xlabel('Predicted Value')
# plt.ylabel('Mean Squared Error')

b, m = polyfit(y_predicted, y_expected, 1)
plt.scatter(y_predicted, y_expected)
plt.plot(np.array(y_predicted), b + m * np.array(y_predicted), '-', color = 'k')

plt.show()

# ============================ non-linear regression 2 ============================

svr_rbf = SVR(kernel = "rbf", C = 100, gamma = 0.1, epsilon = 0.1)

svr_rbf.fit(X_train, Y_train)

y_predicted = []
y_expected = []

for i in range(0, len(X_test)):
	pred = svr_rbf.predict([X_test[i]])

	y_predicted.append(np.round(pred[0], 3))
	y_expected.append(np.round(Y_test[i], 3))

errors = []

_mse = 0

for i in range(len(y_expected)):
	err = (y_expected[i] - y_predicted[i])**2
	errors.append(err)

	_mse += err

print('rbf - MSE: ', np.round(_mse/len(y_expected), 3))
print("Mean squared error: %.2f" % mean_squared_error(y_expected, y_predicted))
# The coefficient of determination: 1 is perfect prediction
print("Coefficient of determination: %.2f\n" % r2_score(y_expected, y_predicted))

plt.title('rbf SVR')
# plt.yticks(np.arange(0.0, 0.6, step = 0.05))
# plt.ylim(0.0, 0.5)

plt.yticks(np.arange(50, 400, step = 50))
plt.xticks(np.arange(50, 400, step = 50))

# plt.plot(errors)
# plt.xlabel('Predicted Value')
# plt.ylabel('Mean Squared Error')

b, m = polyfit(y_predicted, y_expected, 1)
plt.scatter(y_predicted, y_expected)
plt.plot(np.array(y_predicted), b + m * np.array(y_predicted), '-', color = 'k')

plt.show()

# ============================ non-linear regression 3 ============================

DTR = DecisionTreeRegressor(random_state = 0)

DTR.fit(X_train, Y_train)

y_predicted = []
y_expected = []

for i in range(0, len(X_test)):
	pred = DTR.predict([X_test[i]])

	y_predicted.append(np.round(pred[0], 3))
	y_expected.append(np.round(Y_test[i], 3))

errors = []

_mse = 0

for i in range(len(y_expected)):
	err = (y_expected[i] - y_predicted[i])**2
	errors.append(err)

	_mse += err

print('DTR - MSE: ', np.round(_mse/len(y_expected), 3))
print("Mean squared error: %.2f" % mean_squared_error(y_expected, y_predicted))
# The coefficient of determination: 1 is perfect prediction
print("Coefficient of determination: %.2f\n" % r2_score(y_expected, y_predicted))

plt.title('DTR')
# plt.yticks(np.arange(0.0, 0.6, step = 0.05))
# plt.ylim(0.0, 0.5)

plt.yticks(np.arange(50, 400, step = 50))
plt.xticks(np.arange(50, 400, step = 50))

# plt.plot(errors)
# plt.xlabel('Predicted Value')
# plt.ylabel('Mean Squared Error')

b, m = polyfit(y_predicted, y_expected, 1)
plt.scatter(y_predicted, y_expected)
plt.plot(np.array(y_predicted), b + m * np.array(y_predicted), '-', color = 'k')

plt.show()

# ============================ linear regression 2 ============================

linreg = LinearRegression()

linreg.fit(X_train, Y_train)

y_predicted = []
y_expected = []

for i in range(0, len(X_test)):
	pred = linreg.predict([X_test[i]])

	y_predicted.append(np.round(pred[0], 3))
	y_expected.append(np.round(Y_test[i], 3))

errors = []

_mse = 0

for i in range(len(y_expected)):
	err = (y_expected[i] - y_predicted[i])**2
	errors.append(err)

	_mse += err

print('linreg - MSE: ', np.round(_mse/len(y_expected), 3))
print("Mean squared error: %.2f" % mean_squared_error(y_expected, y_predicted))
# The coefficient of determination: 1 is perfect prediction
print("Coefficient of determination: %.2f\n" % r2_score(y_expected, y_predicted))

plt.title('linreg')
# plt.yticks(np.arange(0.0, 0.6, step = 0.05))
# plt.ylim(0.0, 0.5)

plt.yticks(np.arange(50, 400, step = 50))
plt.xticks(np.arange(50, 400, step = 50))

# plt.plot(errors)
# plt.xlabel('Predicted Value')
# plt.ylabel('Mean Squared Error')

b, m = polyfit(y_predicted, y_expected, 1)
plt.scatter(y_predicted, y_expected)
plt.plot(np.array(y_predicted), b + m * np.array(y_predicted), '-', color = 'k')

plt.show()
