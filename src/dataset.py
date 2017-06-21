#!/usr/bin/python
import pandas as pd

class Dataset(object):
    """docstring for Dataset"""
    def __init__(self, filein, fileout):
        super(Dataset, self).__init__()
        self.filein = filein
        self.fileout = fileout

    def readTrainset(self):
        
        tictactoe = pd.read_csv(self.filein,header=None,sep=",").rename(columns={9: "classes"})
        tictactoe["classes"] = tictactoe.apply(lambda r: -1 if r.classes == "negative" else 1,axis=1)
        self.input = tictactoe[[col for col in tictactoe.columns if col !="classes"]]
        self.input = pd.get_dummies(self.input).as_matrix().astype(int)
        self.labels = tictactoe["classes"]
        return self.input, self.labels