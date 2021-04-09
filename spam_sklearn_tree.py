# Finding the optimal number of leaves through cross validation through score
def find_optimal_max_leaves(X_train, Y_train):
    score = []

    for i in range(2, 401):
        print("Iteration " , i)
        classifier = DecisionTreeClassifier(max_leaf_nodes=i)
        score.append(np.sum(cross_val_score(classifier, X_train, Y_train, cv=5))/5)

    index_max = np.argmax(score)
    print(index_max + 2 , " is the best maximum number of leaves in the classifier")
    
    return index_max + 2, score

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier

#Data set setup
dataset = pd.read_csv('Data/spambase.data')
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, -1].values

#Data set split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 3/10, random_state = 699)

#Finding optimal leaves
opt_leaf, Y_axis_cross_val = find_optimal_max_leaves(X_train, Y_train)

#Finding test error for the model with optimal amount of leaves
classifier = DecisionTreeClassifier(max_leaf_nodes = opt_leaf)
classifier.fit(X_train,Y_train)

score = classifier.score(X_test,Y_test)

print("Score for leaves = ", opt_leaf, " is ",  score)

#Plotting Cross Validation Error
X_axis = range(2,401)

plot = plt.plot(X_axis, 1 - np.array(Y_axis_cross_val),'r')
plt.title("Cross Validation Error (Decision Tree)")
plt.xlabel("Number of Leaves")
plt.ylabel("Cross Validation Error")
plt.show()

errors_decision = np.ones(50) * (1 - score)
