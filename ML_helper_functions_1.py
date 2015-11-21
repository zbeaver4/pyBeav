##############################
###Machine Learning Helpers###
##############################

# General libraries.
import collections
import re
import numpy as np
import matplotlib.pyplot as plt
import csv
import datetime
import os

# SK-learn libraries for learning.
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.grid_search import GridSearchCV

# SK-learn libraries for evaluation.
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.metrics import classification_report

def RMSLE(predicted, actual):
	'''Function to calculate the Root Mean Square Logarithmic Error given:
	predicted: list of predicted values
	actual: list of actual values
	'''

  predicted = np.asarray(predicted)
  actual = np.asarray(actual)
  return np.sqrt(
      np.sum(
          np.square(np.log(predicted + 1) - np.log(actual + 1))
      ) / predicted.shape[0])

def plot_all_histograms(dataset, headers = None):
    """Takes in a numpy integer or float matrix and plots the histograms for each feature"""
    
    #Get the number of variables
    feature_count = dataset.shape[1]
    
    #Set the total figure size, giving a 5x5 inch subplot space for each graph
    plt.figure(figsize=(5, dataset.shape[1] * 5))
    
    for i in range(feature_count):
        
        #Get a vector for the feature
        feature_vect = dataset[:, i]
        
        #Move to a new subplot space for each value of k
        plt.subplot(feature_count, 1, i + 1)
        
        #plot a histogram of the feature vector and label it
        plt.hist(feature_vect)
        if headers is not None:
            plt.xlabel(headers[i])
        else:
            plt.xlabel("Variable at index %d" % i)
            
        plt.ylabel("Number of occurrences")
    
    plt.show()
    return

def get_worst(features, n, predicted, actual, headers = None):
    """This function returns the "n" observations from the "features" set
    with the highest squared logarithmic error between the actual and predicted values"""
    
    #Calculate the RMSLE for each observation
    square_log_diff = np.square(np.log(predicted + 1) - np.log(actual + 1))
    
    #Add the square log diff onto the the features array
    all_data = np.append(features, square_log_diff.reshape(-1, 1), 1)
    #(210,9)
    
    #Sort the features array from highest to lowest error
    all_data = all_data[all_data[:,(len(all_data[0]) - 1)].argsort()]
    
    #Add the new column to the headers if headers was passed in and print the headers
    if headers is not None:
        new_headers = headers + ["RMSLE"]
        print "\t".join(new_headers)
    else:
        new_headers = None
        print "\t".join(["index" + str(i) for i in range(features.shape[1])] + ["RMSLE"])
    
    #print the data in a nice format
    print_prettily(all_data[-n:, :])
    
    #plot the histograms for all the data
    plot_all_histograms(all_data[-n:, :], new_headers)
    
    return


def run_classifier(classifier, cl_input, name):
    """This function is the generic function that runs any single sklearn classifier given it
    and produces a corresponding csv file"""
    
    #Create a pipeline to do feature transformation and then run those transformed features through a classifier
    pipeline = Pipeline([
        ('date_split', TimestampTransformer()),
        ('classifier', classifier)
    ])

    # Fit the classifier
    pipeline.fit(cl_input.train_data, cl_input.train_targets)
   
    # Make predictions on dev data
    dev_predictions = pipeline.predict(cl_input.dev_data)
    
    # print dev_predictions, dev_targets
    create_csv_submission(
        './'+name+'/dev_sub.csv', cl_input.dev_data, dev_predictions)

    # Make predictions based on the actual test data.
    predictions = pipeline.predict(cl_input.raw_eval)
    create_csv_submission(
        './'+name+'/eval_sub.csv', cl_input.raw_eval, predictions)
    
    #Return the Root Mean Square Logarithmic Error
    return RMSLE(cl_input.dev_targets, dev_predictions)

def run_rfc_classifier_diagnostics(classifier, cl_input, n, headers = None):
    """This function prints the n worst predictions and their input features along with the
    histogram of these features for a random forest. Calculates error with RMSLE"""
    
    #Pipeline for feature transformation being fed into classifier
    pipeline = Pipeline([
        ('date_split', TimestampTransformer()),
        ('classifier', classifier)
    ])

    # Fit a decision tree (baseline) model on all the data and features
    pipeline.fit(cl_input.train_data, cl_input.train_targets)
   
    # Make predictions on dev data
    dev_predictions = pipeline.predict(cl_input.dev_data)
    
    # print the overall RMSLE
    print "Overall RMSLE is:", RMSLE(cl_input.dev_targets, dev_predictions), "\n"
    
    #print feature importances from random_forest
    if headers is None:
        print "feat_index \t importance" 
        for idx in range(len(classifier.feature_importances_)):
            print idx, "\t", classifier.feature_importances_[idx]
    else:
        print "feat_name \t importance" 
        for name, val in zip(headers, classifier.feature_importances_):
            print name, "\t", val
    
    print
    
    #Return the worse "n" predictions along with histograms of the features
    get_worst(TimestampTransformer().transform(cl_input.dev_data), n, dev_predictions, np.asarray(cl_input.dev_targets), headers)
    return

def run_two_classifiers(classifier1, classifier2, cl_input1, cl_input2, name):
    """This function runs two classifiers on different targets and then adds the results together"""
    
    #Create the pipelines
    pipeline1 = Pipeline([
        ('date_split', TimestampTransformer()),
        ('classifier', classifier1)
    ])
    
    pipeline2 = Pipeline([
        ('date_split', TimestampTransformer()),
        ('classifier', classifier2)
    ])
    
    # Fit a model on all the data and features
    pipeline1.fit(cl_input1.train_data, cl_input1.train_targets)
    pipeline2.fit(cl_input2.train_data, cl_input2.train_targets)
   
    # Make predictions on dev data
    dev_predictions1 = pipeline1.predict(cl_input1.dev_data)
    dev_predictions2 = pipeline2.predict(cl_input2.dev_data)
    tot_dev_predictions = dev_predictions1 + dev_predictions2
    
    # print dev_predictions, dev_targets
    create_csv_submission(
        './'+name+'/dev_sub.csv', cl_input1.dev_data, tot_dev_predictions)

    # Make predictions based on the actual test data.
    test_predict1 = pipeline1.predict(cl_input1.raw_eval)
    test_predict2 = pipeline2.predict(cl_input2.raw_eval)
    tot_test_predict = test_predict1 + test_predict2
    
    create_csv_submission(
        './'+name+'/eval_sub.csv', cl_input1.raw_eval, tot_test_predict)

    return RMSLE(cl_input1.dev_targets, tot_dev_predictions)

# Classifiers
def run_k_neighbors(cl_input):
  k_neighbors = KNeighborsClassifier()
  return run_classifier(k_neighbors, cl_input, 'k_neighbors')

def run_decision_tree(cl_input):
  dec_tree = DecisionTreeClassifier(min_samples_leaf=10, criterion='entropy')
  return run_classifier(dec_tree, cl_input, 'dec_tree')

def run_gaussian_nb(cl_input):
  gaussian_nb = GaussianNB()
  return run_classifier(gaussian_nb, cl_input, 'gaussian_nb')

def run_multinomial_nb(cl_input):
  multinomial_nb = MultinomialNB()
  return run_classifier(multinomial_nb, cl_input, 'multinomial_nb')

def run_linear_regression(cl_input):
  lin_reg = LinearRegression()
  return run_classifier(lin_reg, cl_input, 'lin_reg')

def run_log_regression(cl_input):
  log_reg = LogisticRegression()
  return run_classifier(log_reg, cl_input, 'log_reg')

def run_rfc(cl_input):
  rfc = RandomForestClassifier(
      n_estimators=100,
      criterion='entropy',
      min_samples_leaf=5
      )
  return run_classifier(rfc, cl_input, 'rfc')

# Regressors

def run_k_neighbors_reg(cl_input):
  k_neighbors = KNeighborsRegressor(n_neighbors=3)
  return run_classifier(k_neighbors, cl_input, 'k_neighbors_reg')

def run_decision_tree_reg(cl_input):
  dec_tree = DecisionTreeRegressor(min_samples_leaf=10)
  return run_classifier(dec_tree, cl_input, 'dec_tree_reg')

def run_rfc_reg(cl_input):
  param_grid = [
      # {'n_estimators': [48, 49, 50, 51, 52],
      #  'min_samples_leaf': [1, 2, 3, 4]
      {'n_estimators': [51],
       'min_samples_leaf': [4]
      }]
  grid_search = GridSearchCV(RandomForestRegressor(), param_grid)
  retval = run_classifier(grid_search, cl_input, 'rfc_reg')
  print "Best params:", grid_search.best_params_
  return retval

def run_two_rfcs(cl_input1, cl_input2):
    rfc1 = RandomForestRegressor(n_estimators=51, min_samples_leaf=4)
    rfc2 = RandomForestRegressor(n_estimators=51, min_samples_leaf=4)
    return run_two_classifiers(rfc1, rfc2, cl_input1, cl_input2, 'two_rfcs')

def run_rfc_reg_diagnostics(cl_input, n, headers = None):
    rf = RandomForestRegressor(n_estimators=51, min_samples_leaf=4)
    return run_rfc_classifier_diagnostics(rf, cl_input, n, headers)