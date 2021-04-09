"""
Created on Thu Oct 15 14:34:16 2020

@author: Xu William
"""

#Training useing least squares analytically
def training(expanded_X, t_train):
    X_train2 = expanded_X
    W = np.dot(np.dot(np.linalg.pinv(np.dot(X_train2.T,X_train2)), X_train2.T),t_train)
    return W

#Computing the expected value
def compute_y(expanded_X, W):
    y = np.dot(W,expanded_X.T)
    return y

#Computing the cost using squared error
def compute_cost(Y, t, num_examples):
    cost = 1/num_examples*np.sum(np.square(Y - t))
    return cost

#k-fold validation
def kf_validation(X_training, T_training, n_splits):
    error = 0
    kf = KFold(n_splits = n_splits)
    for train_index, test_index in kf.split(X1_train):
        X_train, X_test = X_training[train_index], X_training[test_index]
        t_train, t_test = T_training[train_index], T_training[test_index]
        
        W = training(X_train, t_train)
        Y = compute_y(X_test, W)
        Cost = compute_cost(Y, t_test, len(test_index))
        error += Cost
    
    error /= n_splits
    return error

def feature_selection(X_train, T_train):
    all_of_all_sets = []
    all_sets = []
    errors = []
    all_of_errors = []
    latest_set = np.ones((X_train.shape[0],1))
    features = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
    all_features = []
    
    #Iterating trough all features and apply kf validation
    for j in range(13):
        for i in features:
            this_set = np.copy(latest_set)
            this_set = np.append(this_set, np.array([X_train[:,i]]).T, axis=1)
            error = kf_validation(this_set, T_train, 5)
            errors.append(error)
            all_sets.append(this_set)
        
        loc = np.argmin(errors)
        all_of_all_sets.append(all_sets[loc])   #Keeping track of values each iteration
        all_of_errors.append(errors[loc])
        latest_set = all_sets[loc]
        chosen_feature = features.pop(loc)
        all_features.append(chosen_feature)
        print("Round ", str(j + 1), "\nThe errors are:", str(errors), "\nFeature ", str(chosen_feature), " is chosen")
        print("Current features :", all_features)
        errors = []
        all_sets = []
    
    return all_of_all_sets, all_features, all_of_errors

#Generating test errors
def generate_test_errors(all_of_all_sets, all_features, t_train, X_test, t_test):
    Costs = []
    latest_set = np.ones((X_test.shape[0],1))
    
    for i in range(13):
        W = training(all_of_all_sets[i], t_train)
        latest_set = np.append(latest_set, np.array([X_test[:,all_features[i]]]).T, axis=1)
        Y = compute_y(latest_set, W)
        Cost = compute_cost(Y, t_test, len(t_test))
        Costs.append(Cost)
    return Costs

#Basic plotting function given training set 
def plot_training_set(X_train, t_train, title_marker):
    for i in range(13):
        plt.scatter(X_train[:,i+1], t_train)
        plt.title("Feature " + str(i+1) + " " + title_marker)
        plt.show()

#Expansion using results from feature selection
def expansion(all_of_all_sets, all_features, powers):
    temp_powers = [0]
    all_of_all_sets_expanded = []
    for i in range(len(all_features)):
        temp_powers.append(powers[all_features[i]])
        all_of_all_sets_expanded.append(all_of_all_sets[i] ** temp_powers)
        
    return all_of_all_sets_expanded
        
    
    
from sklearn.datasets import load_boston
import numpy as np
import matplotlib.pyplot as plt
import sys
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

np.random.seed(699)
X, t = load_boston(return_X_y=True)


# split data into trainig and test sets
X_train, X_test, t_train, t_test = train_test_split(X, t, test_size = 3/10, random_state = 6)
M = len(X_test) #number rows in test set
N = len(X_train) #number rows in train set

# add dummy in trainig set
new_col=np.ones(N)
X1_train = np.insert(X_train, 0, new_col, axis=1)

# add dummy in test set
new_col=np.ones(M)
X1_test = np.insert(X_test, 0, new_col, axis=1)

#Generating arrays from feature selection
all_of_all_sets, all_features, all_of_errors = feature_selection(X1_train, t_train)

#Generating test errors from results of feature selection
test_errors = generate_test_errors(all_of_all_sets, all_features, t_train, X1_test, t_test)
print("The test errors are: " + str(test_errors))

#Plotting validation and test errors
K_values = [1,2,3,4,5,6,7,8,9,10,11,12,13]
plot = plt.plot(K_values, all_of_errors,'r', K_values, test_errors, 'b')
plt.title("Squared Errors VS K Values")
plt.legend(['K-Fold Validation Error', 'Test Error'])
plt.show()

#Plotting the origional training set for all features
plot_training_set(X1_train, t_train, "")

#Feature powers for expansion
expansion_power1 = [0, 0.25, 0.5, 0.5, 1, 1, 1, 1, 0.25, 1, 1, 0.5, 1, 0.25]
expansion_power2 = [0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]

#Expanding features
X_expanded1_train = expansion(all_of_all_sets, all_features, expansion_power1)
X_expanded2_train = expansion(all_of_all_sets, all_features, expansion_power2)

#K-fold validation and errors
kf_error1 = []
kf_error2 = []

for i in X_expanded1_train:
    error = kf_validation(i, t_train, 5)
    kf_error1.append(error)

for i in X_expanded2_train:
    error = kf_validation(i, t_train, 5)
    kf_error2.append(error)
    
print("The validation error for expansion power 1 are: ", kf_error1)
print("The validation error for expansion power 2 are: ", kf_error2)

#Expanding test set
X_test_expanded = X1_test ** expansion_power1

#Calculating Test error
test_errors_expanded = generate_test_errors(X_expanded1_train, all_features, t_train, X_test_expanded, t_test)
print("The test errors are: ", test_errors_expanded)

#Plotting errors
plot = plt.plot(K_values, kf_error1,'r', K_values, test_errors_expanded, 'b')
plt.title("Squared Errors VS K Values For Expanded Set")
plt.legend(['K-Fold Validation Error', 'Test Error'])
plt.show()
