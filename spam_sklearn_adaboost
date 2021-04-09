#Training adaboost with decision tree max depth = 1
def adaboost_depth_1(X_train, X_test, Y_train, Y_test):
    #Training Model
    scores = []
    estimators = []
    for i in range(1, 51):
        classifier = AdaBoostClassifier(base_estimator = DecisionTreeClassifier(max_depth=1),n_estimators=i*50, random_state=0).fit(X_train, Y_train)
        score = classifier.score(X_test, Y_test)
        print("Estimators = ", 50*i, " Score = ", score)
        scores.append(score)
        estimators.append(i*50)
    
    #Calculating Error
    errors = 1 - np.array(scores)
    
    #Plotting Error
    plot = plt.plot(estimators, errors,'r')
    plt.title("Misclassification rate VS Number of Estimators (Adaboost max_depth = 1)")
    plt.xlabel("Number of Estimators")
    plt.ylabel("Misclassification rate")
    plt.show()
    
    return errors

def adaboost_max_node_10(X_train, X_test, Y_train, Y_test):
    #Training Model
    scores = []
    estimators = []
    for i in range(1, 51):
        classifier = AdaBoostClassifier(base_estimator = DecisionTreeClassifier(max_leaf_nodes=10),n_estimators=i*50, random_state=0).fit(X_train, Y_train)
        score = classifier.score(X_test, Y_test)
        print("Estimators = ", 50*i, " Score = ", score)
        scores.append(score)
        estimators.append(i*50)
    
    #Calculating Error
    errors = 1 - np.array(scores)
    
    #Plotting Error
    plot = plt.plot(estimators, errors,'r')
    plt.title("Misclassification rate VS Number of Estimators (Adaboost max_leaf_nodes = 10)")
    plt.xlabel("Number of Estimators")
    plt.ylabel("Misclassification rate")
    plt.show()
    
    return errors

def adaboost_no_restr(X_train, X_test, Y_train, Y_test):
    #Training Model
    scores = []
    estimators = []
    for i in range(1, 51):
        classifier = AdaBoostClassifier(base_estimator = DecisionTreeClassifier(),n_estimators=i*50, random_state=699).fit(X_train, Y_train)
        score = classifier.score(X_test, Y_test)
        print("Estimators = ", 50*i, " Score = ", score)
        scores.append(score)
        estimators.append(i*50)
    
    #Calculating Error
    errors = 1 - np.array(scores)
    
    #Plotting Error
    plot = plt.plot(estimators, errors,'r')
    plt.title("Misclassification rate VS Number of Estimators (Adaboost no restrictions)")
    plt.xlabel("Number of Estimators")
    plt.ylabel("Misclassification rate")
    plt.show()
    
    return errors

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier

#Data set setup
dataset = pd.read_csv('Data/spambase.data')
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, -1].values

#Data set split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 3/10, random_state = 699)

#Training models + Plotting
errors_ada_depth_1 = adaboost_depth_1(X_train, X_test, Y_train, Y_test)
errors_ada_max_node_10 =adaboost_max_node_10(X_train, X_test, Y_train, Y_test)
errors_ada_no_restr =adaboost_no_restr(X_train, X_test, Y_train, Y_test)

#X_ax = list(range(50, 2501, 50))
#plot = plt.plot(X_ax, errors_decision,'r', X_ax, errors_bag, 'b',X_ax, errors_forest, 'k',X_ax, errors_ada_depth_1, 'g',X_ax, errors_ada_max_node_10, 'c',X_ax, errors_ada_no_restr, 'm')
#plt.title("Misclassification Rates")
#plt.legend(['Decision Tree', 'Bagging', 'Random Forest', 'Adaboost Max Depth 1', 'Adaboost Max Nodes 10', 'Adaboost No Restrictions'])
#plt.xlabel("Number of Estimators")
#plt.ylabel("Misclassification rate")
#plt.show()
