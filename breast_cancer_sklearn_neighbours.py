def standardize_features(X_train, X_test):
    num_examples = X_train.shape[0]
    mean = np.sum(X_train, axis = 0)/num_examples
    variance =np.sqrt(1/(num_examples-1)*np.sum(np.square(X_train-mean), axis = 0))
    X_train_norm = (X_train - mean)/variance
    X_test_norm = (X_test - mean)/variance
    return X_train_norm, X_test_norm

# Cross Validation implementation
def kf_validation(X_training, T_training, n_splits, K):
    error = 0
    kf = KFold(n_splits = n_splits)
    for train_index, test_index in kf.split(X_training):
        X_train, X_test = X_training[train_index], X_training[test_index]
        t_train, t_test = T_training[train_index], T_training[test_index]
    
        Y_test = []
        for i in range(len(X_test)):
            Y_test.append(predict(X_train, X_test[i], t_train, K))
        
        Cost = np.sum(np.square(np.array(Y_test)-t_test))/len(t_test)
        error += Cost
    
    error /= n_splits
    return error

# Cross Validation implementation for SKlearn
def kf_validation_sklearn(X_training, T_training, n_splits, K):
    error = 0
    kf = KFold(n_splits = n_splits)
    for train_index, test_index in kf.split(X_training):
        X_train, X_test = X_training[train_index], X_training[test_index]
        t_train, t_test = T_training[train_index], T_training[test_index]
        
        neigh = KNeighborsClassifier(n_neighbors=K)
        neigh.fit(X_train, t_train)
        Cost = 1 - neigh.score(X_test, t_test)
        error += Cost
    
    error /= n_splits
    return error

# Predicting label using training set
def predict(X_train, X_test_individual, Y_train, K):
    distance = []
    for i in range(len(X_train)):
        distance.append(np.sqrt(np.sum(np.square(X_train[i,:] - X_test_individual))))
        
    index = np.argpartition(np.array(distance), K)
    values = index[:K]
    total_count = 0
    for j in values:
        total_count += Y_train[j]
    
    average = total_count/K

    if(average >= 0.5):
        return 1
    else:
        return 0
    
    
from sklearn.datasets import load_breast_cancer
import numpy as np
import matplotlib.pyplot as plt
import sys
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

#Loading data
np.random.seed(699)
X, Y = load_breast_cancer(return_X_y=True)

X_train, X_test, t_train, t_test = train_test_split(X, Y, test_size = 3/10, random_state = 6)

X_train_norm, X_test_norm = standardize_features(X_train, X_test)

#Cross validation for K
for i in range(1,6):
    error = kf_validation(X_train_norm, t_train, 10, i)
    print("The error when K = " , i , " is ", error)

#Predicting test set labels with training data
K = 5
Y_test = []
misclass_rate = 0
for i in range(len(X_test_norm)):
    Y_test.append(predict(X_train_norm, X_test_norm[i], t_train, K))

#Calculaing misclassification rate
misclass_rate = np.sum(np.square(np.array(Y_test)-t_test))/len(t_test)
print("Chosen K = ", K, " The misclassification on the test set is ", misclass_rate)


#----------------------------------SKLearn---------------------------------------
from sklearn.neighbors import KNeighborsClassifier

#Cross Validation for K
for j in range(1,6):
    error = kf_validation_sklearn(X_train_norm, t_train, 10, j)
    print("SKlearn: The error when K = " , j , " is ", error)

#Training model
K = 5
neigh = KNeighborsClassifier(n_neighbors=K)
neigh.fit(X_train_norm, t_train)

#Calculating Misclassification rate
misclass_rate_sklearn = 1 - neigh.score(X_test_norm, t_test)
print("SKlearn: Chosen K = ", K, " The misclassification on the test set is ", misclass_rate_sklearn)
