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


__author__ = 'Michael-Ken Okolo, '
__email__ = 'michaelken.okolo1@csu.fullerton.edu, '
__maintainer__ = 'michaelkenokolo'


def gradient_descent(x_data, y_data):
    m_current = b_current = 0
    iterations = 1000
    n = len(x_data)
    learning_rate = 0.001

    for i in range(iterations):
        y_predicted = m_current * x_data + b_current
        md = -(2 / n) * sum(x_data * y_data - y_predicted)
        bd = -(2 / n) * sum(y_data - y_predicted)
        m_current = m_current - learning_rate * md
        b_current = b_current - learning_rate * bd
        print("m {}, b {}, iteration {}".format(m_current, b_current, i))


if __name__ == '__main__':
    main()
