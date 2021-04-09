#Expanding X to their power depending on M
def expansion(X,M):
    arr = []
    temp = []
    for i in X:
        for j in range(M+1):
            temp.append(i**j)
        arr.append(temp)
        temp = []
    return np.array(arr)
    
#Analytical method using least squares
def training(expanded_X, t_train, M):
    X_train2 = expanded_X
    W = np.dot(np.dot(np.linalg.pinv(np.dot(X_train2.T,X_train2)), X_train2.T),t_train)
    return W

#Computing the expected value
def compute_y(expanded_X, W):
    y = np.dot(W,expanded_X.T)
    return y

#Computing the cost using squared error
def compute_cost(Y, t):
    cost = 1/10*np.sum(np.square(Y - t))
    return cost

#Similar to the training function above except adding regularization
def training_regularization(expanded_X, t_train, M, lambda_B):
    X_train2 = expanded_X
    B = np.zeros((10, 10))
    np.fill_diagonal(B, 2*lambda_B)
    W = np.dot(np.dot(np.linalg.pinv(np.dot(X_train2.T,X_train2) + M/2*B), X_train2.T),t_train)
    return W
# Main program starts here
def train_0_to_8():
    X_train = np.linspace(0.,1.,10) # training set
    X_valid = np.linspace(0.,1.,100) # validation set
    np.random.seed(699)
    t_valid = np.sin(4*np.pi*X_valid) + 0.3 * np.random.randn(100)
    t_train = np.sin(4*np.pi*X_train) + 0.3 * np.random.randn(10)

    X_true = np.linspace(0.,1.,10000)    #Computing true values
    t_true = np.sin(4*np.pi*X_true)
    
    X_true2 = np.linspace(0.,1.,100)    #Computing true values instead with 100
    t_true2 = np.sin(4*np.pi*X_true2)
    
    train_error = []
    valid_error = []
    valid_vs_real_error = []
    for M in range(10):
        expanded_X_train = expansion(X_train, M)    #Expansion for x
        expanded_X_valid = expansion(X_valid, M)

        W = training(expanded_X_train, t_train, M)        #Training W

        Y_train = compute_y(expanded_X_train, W)      #Computing Y value
        Y_valid = compute_y(expanded_X_valid, W)

        cost_train = compute_cost(Y_train, t_train)  #Computing cost
        cost_valid = compute_cost(Y_valid, t_valid)
        
        train_error.append(cost_train)  #Keeping track of cost values
        valid_error.append(cost_valid)
        
        print("M =" , M, " Train Cost =", cost_train, " Valid Cost =", cost_valid)

        #Plotting Y vs X
        plot = plt.plot(X_train, Y_train,'r', X_valid, Y_valid, 'b',X_true, t_true, 'k')
        plt.title("M = " + str(M) + " No Regularization")
        plt.legend(['Training Set', 'Validation Set', 'Real Set'])
        plt.show()
        
        cost_valid_vs_real = compute_cost(Y_valid, t_true2)    #Computing error between valid and real results
        valid_vs_real_error.append(cost_valid_vs_real)
    
    #This is for graphing the average error
    M_values = [0,1,2,3,4,5,6,7,8,9]
    plot = plt.plot(M_values, train_error,'r', M_values, valid_error, 'b',M_values, valid_vs_real_error, 'k')
    plt.title("Squared Error VS M Values")
    plt.legend(['Squared Train Error', 'Squared Valid Error', 'Squared Valid VS Real Error'])
    plt.show()
        
def train_9_with_reg():
    lambda_to_try = [0.00000000001, 0.0000000001, 0.000000001, 0.00000001, 0.0000001, 0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1]
    X_train = np.linspace(0.,1.,10) # training set
    X_valid = np.linspace(0.,1.,100) # validation set
    np.random.seed(699)
    t_valid = np.sin(4*np.pi*X_valid) + 0.3 * np.random.randn(100)
    t_train = np.sin(4*np.pi*X_train) + 0.3 * np.random.randn(10)

    X_true = np.linspace(0.,1.,10000)    #Computing true values
    t_true = np.sin(4*np.pi*X_true)
    
    M = 9
    expanded_X_train = expansion(X_train, M)    #Expansion for x
    expanded_X_valid = expansion(X_valid, M)
    
    for i in lambda_to_try:
        W = training_regularization(expanded_X_train, t_train, M, i)        #Training W

        Y_train = compute_y(expanded_X_train, W)      #Computing Y value
        Y_valid = compute_y(expanded_X_valid, W)

        cost_train = compute_cost(Y_train, t_train)  #Computing cost
        cost_valid = compute_cost(Y_valid, t_valid)

        print("M = " , M, " lambda = ", str(i), "Train Cost =", cost_train, " Valid Cost =", cost_valid)


        plot = plt.plot(X_train, Y_train,'r', X_valid, Y_valid, 'b',X_true, t_true, 'k')
        plt.title("M = " + str(M) + " with Lambda = " + str(i))
        plt.legend(['Training Set', 'Validation Set', 'Real Set'])
        plt.show()
        
        
#Main starts here!!!!
import numpy as np
import matplotlib.pyplot as plt
import sys

train_0_to_8()
#train_9_with_reg()
sys.exit()
