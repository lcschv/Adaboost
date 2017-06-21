#!/usr/bin/python
import math
import time
import numpy as np
from src.weaklearner import DecisionStumps
from sklearn.base import BaseEstimator, ClassifierMixin

class AdaBoost(BaseEstimator,ClassifierMixin):

    def __init__(self, n_estimators):
        self.weights = []
        self.RULES = []
        self.ALPHA = []
        self.n_estimators = n_estimators
        
    def fit(self,X,y):
        #All instances have equal initial weight
        self.weights = np.ones(len(X))/len(X)

        for i in range(self.n_estimators):          
            stumps = DecisionStumps()            
            self.RULES.append(stumps.getRule(X,y,self.weights))
            prediction = stumps.bestCut(X)

            errors = prediction != y
            error = errors.dot(self.weights) 

            self.ALPHA.append(0.5 * (np.log((1 - error)/error)))
            w = self.weights * np.exp(-self.ALPHA[i] * prediction * y)
            self.weights = (w/w.sum()).as_matrix()

    def predict(self, x):
        NR = len(self.RULES)
        hx = [self.ALPHA[i]*self.RULES[i].bestCut(x) for i in range(NR)]
        return np.sign(sum(hx))


