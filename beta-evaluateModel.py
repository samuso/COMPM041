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
	if len(variables) > 0:
		for j in range(len(variables[0])):
			data_points.append([variables[i][j] for i in range(len(variables))])

	data_points = PCA(n_components=7).fit_transform(data_points)
	return data_points, Y

def get_estimates(train_file_name, validate_file_name, continuous_indexes, categorical_indexes, label_index, model):
	indexes = continuous_indexes + categorical_indexes + [label_index]
	X_train, Y_train = get_data(train_file_name, indexes, categorical=categorical_indexes)
	print "imported data from " + train_file_name
	X_validate, Y_validate = get_data(validate_file_name, indexes, categorical=categorical_indexes)
	print "imported data from " + validate_file_name
	model.fit(X_train, Y_train)
	print "trained model"
	predictions = model.predict(X_validate)
	print "predicted values"
	return predictions, Y_validate

def mse(prediction, validation):
	s = 0
	l = len(prediction)
	for i in range(l):
		s += (prediction[i]-validation[i]) ** 2
	return float(s/l)

def analyse_performance(train_file_name,validate_file_name,continuous_indexes,categorical_indexes,label_click,label_bid):
	predicted_bids, actual_bids = get_estimates(train_file_name, validate_file_name, continuous_indexes, categorical_indexes, label_bid, LinearRegression())
	bids_mse = mse(predicted_bids, actual_bids)
	predicted_clicks, actual_click = get_estimates(train_file_name, validate_file_name, continuous_indexes, categorical_indexes, label_click, LinearRegression())
	clicks_mse = mse(predicted_clicks, actual_click)

	trimed_actual_bids = get_clicks(actual_click,actual_bids)
	trimed_predicted_bids = get_clicks(actual_click,predicted_bids)
	trimed_predicted_clicks = get_clicks(actual_click,predicted_clicks)
	min_pctr = sorted(trimed_predicted_clicks)[int(len(trimed_predicted_clicks)*3/4)]

	placed_bids = []
	for i in range(len(trimed_predicted_clicks)):
		if trimed_predicted_clicks[i]>min_pctr:
			placed_bids.append(trimed_predicted_clicks[i]*trimed_predicted_bids[i]*1333)
		else:
			placed_bids.append(trimed_predicted_clicks[i]*trimed_predicted_bids[i]*1333)

	print len(placed_bids)
	print sum([1 for i in range(len(placed_bids)) if placed_bids[i] > trimed_actual_bids[i]])
	print bids_mse
	print clicks_mse
	print '*****************************'

def trim_to(list_to_trim, inclusion_vector):
	trimmed_list = []
	for i in range(len(inclusion_vector)):
		if inclusion_vector[i]:
			trimmed_list.append(list_to_trim[i])
	return trimmed_list

def create_inclusion_vector(length, base_10number):
	inclusion_vector = []
	n = base_10number
	for i in range(length):
		inclusion_vector.append(True if n%2==0 else False)
		n /= 2
	return inclusion_vector

def create_indexes(continuous_indexes,categorical_indexes,n):
	all_indexes = continuous_indexes + categorical_indexes
	trimed_indexes = trim_to(all_indexes,create_inclusion_vector(len(all_indexes), n))
	trimmed_continuous_indexes = [x for x in trimed_indexes if x in continuous_indexes]
	trimmed_categroical_indexes = [x for x in trimed_indexes if x in categorical_indexes]
	return trimmed_continuous_indexes, trimmed_categroical_indexes


train_file_name = 'train.csv'
validate_file_name = 'validation.csv'
# continuous_indexes = [15,16,19,22]
# categorical_indexes = [1,2,6,8,9,10,17,18,20,23,24,25]
continuous_indexes = [16]
categorical_indexes = [24,2,8]

for i in range(1):
	trimmed_continuous_indexes, trimmed_categroical_indexes = create_indexes(continuous_indexes,categorical_indexes,i)
	print create_indexes(continuous_indexes,categorical_indexes,i)
	analyse_performance(train_file_name,validate_file_name, trimmed_continuous_indexes, trimmed_categroical_indexes, 0,21)














