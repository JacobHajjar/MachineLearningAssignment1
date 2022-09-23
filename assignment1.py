#!/usr/bin/env python3
import sys
import pandas as pd
import numpy
from sklearn import linear_model

''' develop the best predictive model based on the chemical engineering dataset'''

__author__ = 'Jacob Hajjar, '
__email__ = 'hajjarj@csu.fullerton.edu, '
__maintainer__ = 'jacobhajjar'

def main():
    '''the main function'''
    data_frame = pd.read_csv("Data1.csv")
    x_data = data_frame[["T", "P", "TC", "SV"]].to_numpy()
    y_data = data_frame["Idx"].to_numpy()

    reg = linear_model.LinearRegression()
    

if __name__ == '__main__':
    main()