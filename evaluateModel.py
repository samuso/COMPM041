import pandas
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas
import statsmodels.api as sm
import scipy

# For 3d plots. This import is necessary to have 3D plotting below
from mpl_toolkits.mplot3d import Axes3D

# For statistics. Requires statsmodels 5.0 or more
from statsmodels.formula.api import ols
# Analysis of Variance (ANOVA) on linear models
from statsmodels.stats.anova import anova_lm
from sklearn.linear_model import LogisticRegression

def get_data(file_path, col_indexes, limit=-1):
	data = []
	counter = 0
	with open(file_path,"r") as input_file:
	    for line in input_file:
	        if counter > 1:
	            data.append(line[:-1].split(','))
	        counter += 1
	print len(data)

	limit = len(data) if limit == -1 else limit

	Xs = []
	Y = []
	for i in col_indexes:
	    if i  == col_indexes[len(col_indexes) - 1]:
	        Y = [int(x[i]) for x in data]
	    else:
	        Xs.append([x[i] for x in data])

	variables = []
	for i in range(len(col_indexes) - 1):
		variables.append([int(Xs[i][j]) for j in range(limit)])
	Y = Y[:limit]

	variables = [scipy.stats.zscore(variable) for variable in variables]

	return variables, Y

indexes = [15,16,19,21]
variables, Y = get_data('train_10percent.csv', indexes)
price_paid = Y
data = pandas.DataFrame({'x1': variables[0], 'x2' : variables[1], 'x3' : variables[2], 'y': Y})
model_price = ols("y ~ x1 + x2 + x3", data).fit()
a = model_price.predict(data)

indexes = [15,16,19,0]
# indexes = [14,15,18,0] # for test
variables, Y = get_data('train_10percent.csv', indexes)
data = pandas.DataFrame({'x1': variables[0], 'x2' : variables[1], 'x3' : variables[2], 'y': Y})
model_clicks = ols("y ~ x1 + x2 + x3", data).fit()
print model_clicks.summary()

variables, Y = get_data('train_10percent.csv', indexes)
data = pandas.DataFrame({'x1': variables[0], 'x2' : variables[1], 'x3' : variables[2], 'y': Y})
b = model_clicks.predict(data)

def get_clicks(Y, list_to_trim):
	clicks = [i for i in range(len(Y)) if Y[i] == 1]
	return [list_to_trim[i] for i in clicks]

a = get_clicks(Y,a)
b = get_clicks(Y,b)

final_bids = [a[i]*b[i]*1333 for i in range(len(a))]
actual_bids = get_clicks(Y,price_paid)

print len(final_bids)
print sum([1 for i in range(len(final_bids)) if final_bids[i] > actual_bids[i]])













