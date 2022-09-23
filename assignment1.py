#!/usr/bin/env python3
import sys
import pandas as pd
from sklearn import linear_model

''' develop the best predictive model based on the chemical engineering dataset'''

__author__ = 'Jacob Hajjar, '
__email__ = 'hajjarj@csu.fullerton.edu, '
__maintainer__ = 'jacobhajjar'

def main():
    '''the main function'''
    data_file = pd.read_csv("Data1.csv")
    reg = linear_model.LinearRegression()
    

if __name__ == '__main__':
    main()