#!/usr/bin/python
import math
import time
import numpy as np
from src.weaklearner import DecisionStumps
from sklearn.base import BaseEstimator, ClassifierMixin

#Used the format of the scikit classifiers, fit and predict, that way
# I was abble to use cross_val from scikit
class AdaBoost(BaseEstimator,ClassifierMixin):

    def __init__(self, n_iterations):
        self.RULES = []
        self.weights = []
        self.n_iterations = n_iterations
        self.ALPHA = []
        
    def fit(self,X,y):
        #Create the weight array
        self.weights = np.ones(len(X))/len(X)
        for i in range(self.n_iterations):          
            stumps = DecisionStumps()            
            self.RULES.append(stumps.getRule(X,y,self.weights))
            if stumps.onecut ==1:
                #If all one is the best cut
                prediction = np.ones(len(X))
            elif stumps.onecut == -1:
                #If all -1 is the best cut
                prediction = np.ones(len(X))*-1
            else:
                #If all one is the best cut is a different value
                prediction = stumps.bestCut(X)

            #get errors
            errors = prediction != y
            #multiply errors by the weights
            error = np.dot(errors,self.weights)
            ##Store Alpha value
            self.ALPHA.append(0.5 * (np.log((1 - error)/error)))
            #update weights
            w = self.weights * np.exp(-self.ALPHA[i] * prediction * y)
            #Normalize weights
            self.weights = (w/w.sum()).as_matrix()

    #predict function used to get the real error
    def predict(self, x):
        NR = len(self.RULES)
        hx = [self.ALPHA[i]*self.RULES[i].bestCut(x) for i in range(NR)]
        #return the sign of the sum of all weak classifiers
        return np.sign(sum(hx))


