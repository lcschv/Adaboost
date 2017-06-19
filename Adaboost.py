#!/usr/bin/python
import argparse
import sys
# from src.input import Input
from src.decisionstumps import DecisionStumps

__author__ = 'Lucas Chaves'


def get_args():
	#Description of the program
	parser = argparse.ArgumentParser(description='Realize the hand-written digits recognition.')
	#Add arguments to the program
	parser.add_argument(
		"-i", "--input", help="Directs the input to the dataset of your choice.", required=True)
	parser.add_argument(
		"-o", "--output", help="Directs the output to a name of your choice.", required=True)
	#Array containing all the prguments passed to the program
	args = parser.parse_args()
	
	#Assign args to variables
	input_file = args.input
	output_file = args.output
	
	#return variables
	return input_file, output_file


def main():
	filein, fileout = get_args()
	
	decisionstump = DecisionStumps(filein, fileout)

if __name__ == "__main__":
	main()