#!/usr/bin/env python3
import sys
import time
import math
import pandas as pd
import numpy
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score

''' develop the best predictive model based on the chemical engineering dataset'''

__author__ = 'Jacob Hajjar, Michael-Ken Okolo'
__email__ = 'hajjarj@csu.fullerton.edu, michaelken.okolo1@csu.fullerton.edu'
__maintainer__ = 'jacobhajjar, michaelkenokolo'


def main():
    '''the main function'''
    data_frame = pd.read_csv("Data1.csv")
    x_data = data_frame[["T", "P", "TC", "SV"]].to_numpy()
    y_data = data_frame["Idx"].to_numpy()
    least_squares_with_libraries(x_data, y_data)

def least_squares_with_libraries(x_data, y_data):

    testing_separation_index = math.floor(len(x_data) * 0.8)

    #separate 80% of the data to training
    x_training = x_data[:testing_separation_index] 
    x_testing = x_data[testing_separation_index:] 

    y_training = y_data[:testing_separation_index]
    y_testing = y_data[testing_separation_index:] 

    #perform least squares regression
    reg = linear_model.LinearRegression()
    starting_time = time.time()
    reg.fit(x_training, y_training)
    finishing_time = time.time()
    elapsed_time = finishing_time - starting_time
    print(elapsed_time)
    print(reg.coef_)

    #predict new values 
    y_predicted = reg.predict(x_testing)

    print("The root mean squared error is", mean_squared_error(y_testing, y_predicted))
    print("The r squared score is", r2_score(y_testing, y_predicted))


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
