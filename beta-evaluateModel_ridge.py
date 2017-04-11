import pandas
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas
import statsmodels.api as sm
import scipy
from sklearn.decomposition import PCA

# For 3d plots. This import is necessary to have 3D plotting below
from mpl_toolkits.mplot3d import Axes3D

# For statistics. Requires statsmodels 5.0 or more
from statsmodels.formula.api import ols
# Analysis of Variance (ANOVA) on linear models
from statsmodels.stats.anova import anova_lm
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
from sklearn import svm
from sklearn.preprocessing import OneHotEncoder

categorical_data_mapping = {}

def update_category_to_value_mapping(values, name):
	if name in categorical_data_mapping:
		pass
	else:
		categorical_data_mapping[name] = {}
		length = len(values)
		mapped_to = 0
		for e in values:
			a = [0 for i in range(mapped_to)] + [1] + [0 for i in range(length-mapped_to-1)]
			categorical_data_mapping[name][e] = a
			mapped_to += 1

def get_value_for(key, name):
	if key not in categorical_data_mapping[name]:
		for a in categorical_data_mapping[name]:
			return [0 for i in range(len(a))]
	else:
		return categorical_data_mapping[name][key]

def get_clicks(Y, list_to_trim):
	clicks = [i for i in range(len(Y)) if Y[i] == 1]
	return [list_to_trim[i] for i in clicks]

def get_data(file_path, col_indexes, limit=-1, categorical=[]):
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
	    	if i in categorical:
	    		update_category_to_value_mapping(set([x[i] for x in data]),str(i))
	    		encodings = [get_value_for(x[i], str(i)) for x in data]
	    		for j in range(len(encodings[0])):
	    			feature = [encodings[k][j] for k in range(len(encodings))]
	    			Xs.append(feature)
	    	else:
	        	Xs.append([x[i] for x in data])

	variables = []
	for i in range(len(Xs)):
		variables.append([int(Xs[i][j]) for j in range(limit)])
	Y = Y[:limit]

	data_points = []
	for j in range(len(variables[0])):
		data_points.append([variables[i][j] for i in range(len(variables))])

	# data_points = PCA(n_components=7).fit_transform(data_points)
	return data_points, Y

# categorical_indexes = [10,17,18,24]
categorical_indexes = [17,18,24]
continuous_indexes = [1,2,15,16,19]
indexes1 = continuous_indexes + categorical_indexes + [21]
data_points, price_paid = get_data('train.csv', indexes1, categorical=categorical_indexes)

regr_c = Ridge(alpha=1.)
regr_c.fit(data_points, price_paid)
# regr_c = svm.SVR(kernel='rbf', C=1)
# regr_c.fit(data_points, price_paid)

indexes2 = continuous_indexes + categorical_indexes + [0]
data_points, clicked = get_data('train.csv', indexes2, categorical=categorical_indexes)

regr_p = Ridge(alpha=1.)
regr_p.fit(data_points, clicked)

data_points_to_predict, actual_bids = get_data('validation.csv', indexes1, categorical=categorical_indexes)
data_points_to_predict, clicked = get_data('validation.csv', indexes2, categorical=categorical_indexes)

a = regr_c.predict(data_points_to_predict)
b = regr_p.predict(data_points_to_predict)
print 'sum of all possibilities = ' + str(sum(b))

actual_bids = get_clicks(clicked,actual_bids)
a = get_clicks(clicked,a)
b = get_clicks(clicked,b)
final_bids = [a[i]*b[i]*1333 for i in range(len(a))]

print len(final_bids)
print sum([1 for i in range(len(final_bids)) if final_bids[i] > actual_bids[i]])
print a[:10]
print b[:10]














