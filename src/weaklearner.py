import numpy as np

class DecisionStumps():

    def __init__(self):     
        self.column_pred = None
        self.inverted = None
        self.constant = False

    def set_rule(self,X_input,y,weight):

        X = X_input.copy()
        best_error = float("inf")

        #stumps that predict the column values
        for c in range(len(X[0])):
            X[X[:,c] == 0,c] = -1
            errors = (X[:,c] != y)
            error = errors.dot(weight)
            if(error < best_error):
                best_error = error
                best_h_column = c
                self.inverted = False

        #stumps that predict the inverse of the column values
        for c in range(len(X[0])):                     
            X[X[:,c] == 1,c] = -1
            X[X[:,c] == 0,c] =  1
            errors = (X[:,c] != y)
            error =errors.dot(weight)            
            if(error < best_error):
                best_error = error
                best_h_column = c
                self.inverted = True

        #constant 1 prediction
        error = (np.ones(X.shape[0]) != y).dot(weight)
        if(error < best_error):
            self.constant = 1
        #constant -1 prediction

        error = (np.ones(X.shape[0])*-1 != y).dot(weight)
        if(error < best_error):
            self.constant = -1

        self.column_pred = best_h_column
        return self

    def bestCut(self,X_input):        
        X = X_input.copy()
        if(self.constant == 1):
            return np.ones(len(X))
        elif(self.constant == -1):
            return np.ones(len(X)) *-1

        if(self.inverted):
            X[X[:,self.column_pred] == 1,self.column_pred] = -1
            X[X[:,self.column_pred] == 0,self.column_pred] =  1
        else:
            X[X[:,self.column_pred] == 0,self.column_pred] = -1
        return X[:,self.column_pred]