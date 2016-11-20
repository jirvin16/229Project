import h5py
import os
import time

import numpy as np
import tensorflow as tf

from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

with h5py.File('data_vectors/data.h5') as hf:
	X    		 	 = hf["X"][:]
	genres 	 		 = hf["y"][:]
	genre_names 	 = hf["genres"][:]
	mean   	 		 = hf["mean"]
	std    	 		 = hf["std"]

random_seed 	= 42
np.random.seed(random_seed)

permutation 	= np.random.permutation(X.shape[0])
shuffled_X 	    = X[permutation]
shuffled_genres = genres[permutation]
X_train 	    = shuffled_X[:650]
y_train 	    = shuffled_genres[:650]
X_valid	        = shuffled_X[650:750]
y_valid   	    = shuffled_genres[650:750]
X_test      	= shuffled_X[750:]
y_test	    	= shuffled_genres[750:]

def create_linear_svm(params):
	if 'penalty' in params:
		penalty = params['penalty']
	else:
		penalty = 'l2'
	if 'loss' in params:
		loss = params['loss']
	else:
		loss = 'squared_hinge'
	if 'dual' in params:
		dual = params['dual']
	else:
		dual = True
	return LinearSVC(penalty=penalty, loss=loss, dual=dual)

def create_svm(params):
	if 'decision_function_shape' in params:
		decision_function_shape = params['decision_function_shape']
	else:
		decision_function_shape = None
	if 'kernel' in params:
		kernel = params['kernel']
	else:
		kernel = 'rbf'
	return SVC(decision_function_shape=decision_function_shape, kernel=kernel)

def create_logistic_regression(params):
	if 'multi_class' in params:
		multi_class = params['multi_class']
	else:
		multi_class = 'ovr'
	if 'solver' in params:
		solver = params['solver']
	else:
		solver = 'liblinear'
	return LogisticRegression(multi_class=multi_class, solver=solver)

linear_params = [{}, {'penalty':'l1', 'dual': False}, {'loss': 'hinge'}]

print("Linear SVM")
for param in linear_params:
	print(param)
	clf = create_linear_svm(param)
	clf.fit(X_train, y_train)
	print("Test Score: ", clf.score(X_test, y_test))

params = [{}, {'decision_function_shape':'ovo'}, {'decision_function_shape':'ovr'},
		  {'decision_function_shape':'ovo', 'kernel': 'poly'},
		  {'decision_function_shape':'ovo', 'kernel': 'sigmoid'},
		  {'decision_function_shape':'ovr', 'kernel': 'poly'},
		  {'decision_function_shape':'ovr', 'kernel': 'sigmoid'}]

print("Nonlinear SVM")
for param in params:
	print(param)
	clf = create_svm(param)
	clf.fit(X_train, y_train)
	print("Test Score: ", clf.score(X_test, y_test))

print("Logistic Regression")
log_params = [{}, {'multi_class':'multinomial', 'solver':'newton-cg'}]
for param in log_params:
	print(param)
	clf = create_logistic_regression(param)
	clf.fit(X_train, y_train)
	print("Test Score: ", clf.score(X_test, y_test))




