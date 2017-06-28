import numpy as np

class DecisionStumps():

    def __init__(self):     
        self.best_cut = None

    def getRule(self,X_input,y,weight):
        X = X_input.copy()
        min_error = 9999.999
        #Calculates every single cut possible
        for c in range(len(X[0])):
            errors = (X[:,c] != y)      
            error = np.dot(errors,weight)
            if(error < min_error):
                min_error = error
                self.best_cut = c

        #Check if setting just true or false to everything is the best cut
        errors = (np.ones(len(X)) != y)
        error = np.dot(errors, weight)
        self.onecut = False
        if(error < min_error and 1-error > 0.5):
            self.onecut = 1
        elif 1-error < min_error:
            self.onecut = -1
        return self


    def bestCut(self,X_input):        
        X = X_input.copy()
        #Check if it was only one cut
        if(self.onecut == 1):
            return np.ones(len(X))
        elif(self.onecut == -1):
            return np.ones(len(X)) *-1
        
        #return the best cut if it's not only one cut
        X[X[:,self.best_cut] == 0,self.best_cut] = -1
        
        return X[:,self.best_cut]