import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier

#Data set setup
dataset = pd.read_csv('Data/spambase.data')
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, -1].values

#Data set split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 3/10, random_state = 699)

#Training models
scores = []
estimators = []
for i in range(1, 51):
    classifier = BaggingClassifier(n_estimators=50*i, random_state=699).fit(X_train, Y_train)
    score = classifier.score(X_test, Y_test)
    print("Estimators = ", 50*i, " Score = ", score)
    scores.append(score)
    estimators.append(i*50)

#Calculating error through score
errors_bag = 1 - np.array(scores)

#plotting misclassification rate
plot = plt.plot(estimators, errors_bag,'r')
plt.title("Misclassification rate VS Number of Estimators (Bagging)")
plt.xlabel("Number of Estimators")
plt.ylabel("Misclassification rate")
plt.show()
