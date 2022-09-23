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

    x_training = x_data[:-20] #beginning of vector minus the last 20
    x_testing = x_data[-20:] #last 20 in the vector

    y_training = y_data[:-20] #beginning of vector minus the last 20
    y_testing = y_data[-20:] #last 20 in the vector

    reg = linear_model.LinearRegression()
    reg.fit(x_training, y_training)
    print(reg.coef_)

if __name__ == '__main__':
    main()