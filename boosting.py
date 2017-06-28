#!/usr/bin/python
import argparse
import sys
from src.adaboost import AdaBoost
from src.adaboost import DecisionStumps
from src.dataset import Dataset
from sklearn.model_selection import cross_val_score
import numpy as np

__author__ = 'Lucas Chaves'

def get_args():
	#Description of the program
	parser = argparse.ArgumentParser(description='Realize the hand-written digits recognition.')
	#Add arguments to the program
	parser.add_argument(
		"-i", "--input", help="Directs the input to the dataset of your choice.", required=True)
	parser.add_argument(
		"-o", "--output", help="Directs the output to a name of your choice.", required=True)
	parser.add_argument(
		"-it", "--iterations", type=int, help="Number of iterations.", required=True)
	parser.add_argument(
		"-k", "--Kfolds", type=int, help="Number of iterations.", required=True)
	
	#Array containing all the arguments passed to the program
	args = parser.parse_args()
	
	#Assign args to variables
	input_file = args.input
	output_file = args.output
	iterations= args.iterations
	Kfolds = args.Kfolds
	
	#return variables
	return input_file, output_file, iterations, Kfolds

def main():
	filein, fileout, iterations, Kfolds = get_args()
	data = Dataset(filein, fileout)
	X, y = data.readTrainset()
	## Each iteration is the number of weaklearners used.
	print "Ite. ----- Accuracy ----- Error--"
	for n_iterations in range(1, iterations):		
		clf = AdaBoost(n_iterations)
		scores = cross_val_score(clf, X, y, cv=Kfolds)
		print(str(n_iterations)+ "	"+str(np.array(scores).mean())+"	"+str(1-np.array(scores).mean()))

if __name__ == "__main__":
	main()
