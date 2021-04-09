# Standardize features with mean of 0 and variance of 1
def standardize_features(X_train, X_test):
    num_examples = X_train.shape[0]
    mean = np.sum(X_train, axis = 0)/num_examples
    variance =np.sqrt(1/(num_examples-1)*np.sum(np.square(X_train-mean), axis = 0))
    X_train_norm = (X_train - mean)/variance
    X_test_norm = (X_test - mean)/variance
    return X_train_norm, X_test_norm

# Initialize weights for log regression
def weight_initialize_logistic(num_of_features):
    W = np.zeros(num_of_features) 
    return W

# Calculating the results before assigning a label
def calculate_y_hat_logistic(X, W):
    Z = np.dot(W, X.T)
    Y_hat = 1/(1 + np.exp(-Z)) 
    return Z, Y_hat

# Computing Cost
def compute_cost_logistic(Y_hat_train, Y_train):
    num_examples = Y_hat_train.shape[0]
    cost = -1/num_examples*np.sum(np.multiply(Y_train,np.log(Y_hat_train))+np.multiply(1-Y_train,np.log(1-Y_hat_train)))
    return cost

# Updating weights for the next epoch
def update_weights_logistic(W, X_train, Y_hat_train, Y_train, alpha = 0.01):
    num_examples = Y_hat_train.shape[0]
    W_new = W - alpha/num_examples*np.dot(X_train.T, Y_hat_train-Y_train)
    return W_new

# Calculating prescieion and recall
def get_precision_recall(Y_hat, Y):
    sorted_array = np.sort(Y_hat)[::-1]
    precision = []
    recall = []
    F1 = []
    
    for i in range(len(sorted_array)):
        results = compute_prediction_logistic(Y_hat, sorted_array[i])
        TP, FP, FN = find_false_true_positive_negative(results, Y)
        precision.append(TP/(TP+FP))
        recall.append(TP/(TP+FN))
        F1.append(2*precision[i]*recall[i]/(precision[i]+recall[i]))
    return precision, recall , F1
    
    
# Assigning labels
def compute_prediction_logistic(Y_hat_train, threshold):
    results = []
    for i in Y_hat_train:
        if(i >= threshold):
            results.append(1)
        else:
            results.append(0)
    
    return results

#Counting True Positive, False Positive and False Negative results
def find_false_true_positive_negative(results, Y):
    TP = 0
    FP = 0
    FN = 0
    
    for i in range(len(results)):
        if(results[i] == 1 and Y[i] == 1):
            TP += 1
        elif(results[i] == 1 and Y[i] == 0):
            FP += 1
        elif(results[i] == 0 and Y[i] == 1):
            FN += 1

    return TP, FP, FN

# Misclassification rate calculation
def get_misclass_rate_logistic(Y_hat, Y):
    misclass_rate = np.sum(np.abs(compute_prediction_logistic(Y_hat, 0.5)-Y))/Y_hat.shape[0]
    return misclass_rate

# F1 score calculation
def get_F1_logistic(Y_hat, Y):
    results = compute_prediction_logistic(Y_hat, 0.5)
    TP, FP, FN = find_false_true_positive_negative(results, Y)
    P = TP/(TP+FP)
    R = TP/(TP+FN)
    F1 = 2*P*R/(P+R)
    return F1

from sklearn.datasets import load_breast_cancer
import numpy as np
import matplotlib.pyplot as plt
import sys
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

#Loading data
np.random.seed(699)
X, Y = load_breast_cancer(return_X_y=True)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 3/10, random_state = 6)

X_train_norm, X_test_norm = standardize_features(X_train, X_test)

# add dummy in train set
new_col=np.ones(len(X_train_norm))
X_train_norm = np.insert(X_train_norm, 0, new_col, axis=1)

# add dummy in test set
new_col=np.ones(len(X_test_norm))
X_test_norm = np.insert(X_test_norm, 0, new_col, axis=1)

# initialize weights
W = weight_initialize_logistic(X_train_norm.shape[1])

# training model
n = 5001
for i in range(n):
    Z, Y_hat_train = calculate_y_hat_logistic(X_train_norm, W)
    cost = compute_cost_logistic(Y_hat_train,Y_train)
    if(i%500 == 0):
        print("The cost after ",i ," epochs is ", cost)
    if(i == n-1):
        break
    W = update_weights_logistic(W, X_train_norm, Y_hat_train, Y_train)

# testing results using trained model
Z, Y_hat_test = calculate_y_hat_logistic(X_test_norm, W)

# Calculating error
misclass_rate = get_misclass_rate_logistic(Y_hat_test, Y_test)
print("Misclassification rate for self logistic regression is ", misclass_rate)

#PR calculation
precision, recall, F1 = get_precision_recall(Y_hat_test, Y_test)

#F1 score
F1 = get_F1_logistic(Y_hat_test, Y_test)
print("F1 Score for self logistic regression is ", F1)

#Plotting PR curve
plot = plt.plot(recall, precision,'r')
plt.title("Precision Recall Curve")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.show()
#--------------------------Sklearn Logistic------------------------------
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import plot_precision_recall_curve
from sklearn.metrics import average_precision_score

#Trining SKlearn model
sci_log = LogisticRegression(random_state=0).fit(X_train_norm, Y_train)
sci_Y_predict = sci_log.predict(X_test_norm)

#Calculating misclassification rate
sci_misclass_rate = 1 - sci_log.score(X_test_norm, Y_test)
print("Misclassification Rate for SKlearn logistic regression is ", sci_misclass_rate)

#Calculating f1 score
sci_F1 = f1_score(Y_test, sci_Y_predict, average='macro')
print("F1 Score for SKlearn logistic regression is ", sci_F1)

#PR curve
sci_y_score = sci_log.decision_function(X_test_norm)
sci_average_precision = average_precision_score(Y_test, sci_y_score)

disp = plot_precision_recall_curve(sci_log, X_test_norm, Y_test)
disp.ax_.set_title('2-class Precision-Recall curve: '
                   'AP={0:0.2f}'.format(sci_average_precision))

print(sci_log.coef_)
