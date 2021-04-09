def sigmoid(X):
    return 1/(1 + np.exp(-X)) 

#relu function
def relu(X):
    return np.maximum(0 ,X)

#sigmoid derivative function
def reluDerivative(x):
     x[x<=0] = 0
     x[x>0] = 1
     return x

#figuring out the size of each layer
def layer_sizes(X, Y, n_h1, n_h2):
    n_x = np.size(X,0)
    n_h1 = n_h1
    n_h2 = n_h2
    n_y = np.size(Y,0)
    
    return n_x, n_h1, n_h2, n_y

#He initialization of weights
def initialize_weights(n_x, n_h1, n_h2, n_y):
    W1 = np.random.randn(n_h1,n_x) * (2/n_x)
    b1 = np.zeros((n_h1,1))
    W2 = np.random.randn(n_h2,n_h1) * (2/n_h1)
    b2 = np.zeros((n_h2,1))
    W3 = np.random.randn(n_y,n_h2) * (2/n_h2)
    b3 = np.zeros((n_y,1))
    
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2,
                  "W3": W3,
                  "b3": b3}
    return parameters

#Neural network forward propogation
def forward_propagation(X, parameters):
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    W3 = parameters["W3"]
    b3 = parameters["b3"]
    
    Z1 = W1.dot(X)+b1
    A1 = relu(Z1)
    Z2 = W2.dot(A1)+b2
    A2 = relu(Z2)
    Z3 = W3.dot(A2)+b3
    A3 = sigmoid(Z3)
    

    
    cache = {"Z1": Z1,
             "A1": A1,
             "Z2": Z2,
             "A2": A2,
             "Z3": Z3,
             "A3": A3}
    
    return A3, cache

#Computing the cost a single iteration
def compute_cost(A3, Y, parameters):
    num = A3.shape[1]
    cost = -1/num*np.sum(np.multiply(np.log(A3),Y)+np.multiply(np.log(1-A3),1-Y))

    return cost

#Finding the gradients
def backward_propagation(parameters, cache, X, Y):

    W1 = parameters["W1"]
    W2 = parameters["W2"]
    W3 = parameters["W3"]
    
    A1 = cache["A1"]
    A2 = cache["A2"]
    A3 = cache["A3"]

    dZ3 = A3 - Y
    dW3 = dZ3.dot(A2.T)
    db3 = dZ3
    dZ2 = np.multiply(W3.T.dot(dZ3), reluDerivative(A2))
    dW2 = dZ2.dot(A1.T)
    db2 = dZ2
    dZ1 = np.multiply(W2.T.dot(dZ2), reluDerivative(A1))
    dW1 = dZ1.dot(X.T)
    db1 = dZ1
    
    gradiants = {"dW1": dW1,
             "db1": db1,
             "dW2": dW2,
             "db2": db2,
             "dW3": dW3,
             "db3": db3}
    
    return gradiants

#Updating the weights based on the gradients
def update_parameters(parameters, gradiants):
    learning_rate = 0.01
    
    W1 = parameters["W1"]
    W2 = parameters["W2"]
    W3 = parameters["W3"]
    b1 = parameters["b1"]
    b2 = parameters["b2"]
    b3 = parameters["b3"]
    
    dW1 = gradiants["dW1"]
    db1 = gradiants["db1"]
    dW2 = gradiants["dW2"]
    db2 = gradiants["db2"]
    dW3 = gradiants["dW3"]
    db3 = gradiants["db3"]
    
    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1
    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2
    W3 = W3 - learning_rate * dW3
    b3 = b3 - learning_rate * db3
    
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2,
                  "W3": W3,
                  "b3": b3}
    
    return parameters

#nerual network implementation
def neural_net(X, Y, n_h1, n_h2, num_epoch, X_valid, Y_valid):
    n_x = np.size(X[:,0],0)
    n_y = 1
    
    iterations = np.size(Y,0)
    parameters = initialize_weights(n_x, n_h1, n_h2, n_y)
    
    costs = []
    valid_costs = []
    misclsss_rate = []
    valid_misclass_rates = []
    
    #Neural network starts running here
    for j in range(0, num_epoch):
        for i in range(0, iterations):
            A3, cache = forward_propagation(np.array([X[:,i]]).T, parameters)
            cost = compute_cost(A3, Y[i], parameters)
            gradiants = backward_propagation(parameters, cache, np.array([X[:,i]]).T, Y[i])
            parameters = update_parameters(parameters, gradiants)
            if(i == 0 and j == 0):
                print("Cost of first iteration is " , cost)
        
        #Cost
        A3_train, cache_train = forward_propagation(X, parameters)
        cost_train = compute_cost(A3_train, Y, parameters)
        print("Cost after epoch ", j+1, " is ", cost)
        costs.append(cost_train)
        
        #Misclassification rate
        mis_rate = misclass_rate(X, Y, parameters)
        misclsss_rate.append(mis_rate)

        #Validation cost
        A3_valid, cache_valid = forward_propagation(X_valid, parameters)
        cost_valid = compute_cost(A3_valid, Y_valid, parameters)
        valid_costs.append(cost_valid)
              
        #Validation Misclassification rate             
        valid_mis_rate = misclass_rate(X_valid, Y_valid, parameters)
        valid_misclass_rates.append(valid_mis_rate)
        
    return parameters, costs, misclsss_rate, valid_costs, valid_misclass_rates

#Making predictions based on results of the sigmoid function
def prediction(parameters, X):
    A3, cache = forward_propagation(X, parameters)
    prediction = []
    
    for i in A3[0]:
        if(i > 0.5):
            prediction.append(1)
        else:
            prediction.append(0)

    return np.array(prediction)

#Finding misclassification rate
def misclass_rate(X, Y, parameters):

    Y_hat = prediction(parameters, X)
    mis_rate = np.sum(np.square(Y_hat - Y)) / len(Y)

    return mis_rate

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import sys

dataset = pd.read_csv('Data/data_banknote_authentication1.txt')
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, -1].values
np.random.seed(699)

# 60/15/15 split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 1.5/10, random_state = 699)
X_train, X_valid, Y_train, Y_valid = train_test_split(X_train, Y_train, test_size = 1.5/8.5, random_state = 699)

sc = StandardScaler()

X_train[:, :]  = sc.fit_transform(X_train[:, :])
X_valid[:, :]  = sc.transform(X_valid[:, :])
X_test[:, :]  = sc.transform(X_test[:, :])

#Working with column vectors
X_train, X_test, Y_train, Y_test, X_valid, Y_valid = X_train.T, X_test.T, Y_train.T, Y_test.T, X_valid.T, Y_valid.T

#Change following parameters
num_feat = 4
n1 = 4
epoch = 100

#deleting rows for other D. please uncomment and change when num_feat != 4
#X_train = np.delete(X_train,(2,3),axis = 0)
#X_test = np.delete(X_test,(2,3),axis = 0)
#X_valid = np.delete(X_valid,(2,3),axis = 0)
#--------------------------

#Runing instances of neural nets and plotting the misclassification rate and cost.
for n2 in range(2,5):
    parameters, costs, misclsss_rate, valid_costs, valid_misclass_rate = neural_net(X_train, Y_train, n1, n2, epoch, X_valid, Y_valid)
    print("Final Validation Misclass Error for n1 = " , n1, " and n2 = ", n2," is ", valid_misclass_rate[-1])
    print("Final Validation Cost for n1 = " , n1, " and n2 = ", n2," is ", valid_costs[-1])
    
    epochs = range(1,101)
    plot = plt.plot(epochs, costs,'r', epochs, valid_costs, 'b')
    plt.title("Costs and Validation Costs for (" + str(n1) + "," + str(n2) + ")")
    plt.legend(['Training Cost', 'Validation Cost'])
    plt.xlabel("Epochs")
    plt.ylabel("Cost")
    plt.show()

    plot = plt.plot(epochs, misclsss_rate,'r', epochs, valid_misclass_rate, 'b')
    plt.title("Misclass Rate and Validation Misclass for (" + str(n1) + "," + str(n2) + ")")
    plt.legend(['Training Misclass Rate', 'Validation Misclass Rate'])
    plt.xlabel("Epochs")
    plt.ylabel("Misclass Rate")
    plt.show()
