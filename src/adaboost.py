#!/usr/bin/python
import numpy as np
import math
import pandas as pd
import time
from numpy import *
class AdaBoost(object):

    def __init__(self, filein, fileout):
        super(AdaBoost, self).__init__()
        self.filein = filein
        self.fileout= fileout
        self.readTrainSet()
        

    def set_rule(self, func, test=False):
        errors = []
        for t in range(len(self.input)):
            if (self.labels[t] != func(self.input[t])):
                errors.append(True)
            else:
                errors.append(False)

        # errors = array([t[1]!=func(t[0]) for t in self.training_set])
        e = (errors*self.weights).sum()
        if test:
            return e
        alpha = 0.5 * log((1-e)/e)
        print 'e=%.2f a=%.2f'%(e, alpha)
        w = zeros(self.N)
        for i in range(self.N):
            if errors[i] == 1: 
                w[i] = self.weights[i] * exp(alpha)
            else: 
                w[i] = self.weights[i] * exp(-alpha)
        self.weights = w / w.sum()
        self.RULES.append(func)
        self.ALPHA.append(alpha)

    def evaluate(self):
        NR = len(self.RULES)
        for (x,l) in zip(self.input,self.labels):
            for i in range(NR):
                hx = [self.ALPHA[i]*self.RULES[i](x)]
            # hx = [self.ALPHA[i]*self.RULES[i](x) for i in range(NR)]
            # print x, sign(l) == sign(sum(hx))
 
    def readTrainSet(self):
        tictactoe = pd.read_csv(self.filein,header=None,sep=",").rename(columns={9:"expected"})
        self.input = tictactoe[[col for col in tictactoe.columns if col !="expected"]]
        # print len(self.input[1])

        tictactoe["expected"] = tictactoe.apply(lambda r: 1 if r.expected == "positive" else -1,axis=1)
        
        # Creating hotencoder for the categorial input.
        # self.input = pd.get_dummies(tictactoe[[c for c in tictactoe.columns if c !="expected"]]).as_matrix().astype(int)
        self.labels = tictactoe["expected"]
        self.input = pd.get_dummies(self.input).as_matrix().astype(int) 
        # print self.labels
        print self.input[0]
        examples = []
        examples.append(((1,  2),1))
        examples.append(((1,  4),1))
        examples.append(((2.5,5.5),1))
        examples.append(((3.5,6.5),1))
        examples.append(((4,5.4),1))
        examples.append(((2,1),-1))
        examples.append(((2,4),-1))
        examples.append(((3.5,3.5),-1))
        examples.append(((5,2),-1))
        examples.append(((5,5.5),-1))

        self.N = len(self.input)
        self.weights = ones(self.N)/self.N
        self.RULES = []
        self.ALPHA = []
        self.set_rule(lambda x: 2*(x[0] > 0.5 and x[1] > 0.5 and x[2] < 0.5)-1)
        self.set_rule(lambda x: 2*(x[0] < 0.5 and x[1] > 0.5 and x[2] < 0.5)-1)
        self.set_rule(lambda x: 2*(x[0] < 0.5 and x[1] < 0.5 and x[2] > 0.5)-1)
        # self.set_rule(lambda x: 2*(x[1] > 0.5)-1)
        # self.set_rule(lambda x: 2*(x[2] > 0.5)-1)
        # self.set_rule(lambda x: 2*(x[3] > 0.5)-1)
        # self.set_rule(lambda x: 2*(x[4] > 0.5)-1)
        # self.set_rule(lambda x: 2*(x[5] > 0.5)-1)
        self.evaluate()

	    