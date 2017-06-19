#!/usr/bin/python
import numpy as np
import math
import pandas as pd
import time

class DecisionStumps(object):

	def __init__(self, filein, fileout):
		super(DecisionStumps, self).__init__()
		self.filein = filein
		self.fileout = fileout
	
		self.readTrainSet()
	

	
	def readTrainSet(self):
		dataset = pd.read_csv(self.filein,header=None,sep=",").rename(columns={9:"expected"})
		
		#Processing  data
		X = pd.get_dummies(dataset[[col for col in dataset.columns if col !="expected"]]).as_matrix().astype(int)
		labels = dataset["expected"]
		print labels
		