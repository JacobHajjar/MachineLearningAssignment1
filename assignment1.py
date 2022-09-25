#!/usr/bin/env python3
import sys
import time
import math
import pandas as pd
import numpy as np
from numpy import random
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

''' develop the best predictive model based on the chemical engineering dataset'''

__author__ = 'Jacob Hajjar, Michael-Ken Okolo'
__email__ = 'hajjarj@csu.fullerton.edu, michaelken.okolo1@csu.fullerton.edu'
__maintainer__ = 'jacobhajjar, michaelkenokolo'


def main():
    """the main function"""
    data_frame = pd.read_csv("Data1.csv")
    x_data = data_frame[["T", "P", "TC", "SV"]].to_numpy()
    y_data = data_frame["Idx"].to_numpy()

    # [80-20] training-testing %
    testing_separation_index = math.floor(len(x_data) * 0.8)

    least_squares_with_libraries(x_data, y_data, testing_separation_index)
    gradient_descent(x_data, y_data, testing_separation_index)


def least_squares_with_libraries(x_data, y_data, testing_separation_index):
    # separate 80% of the data to training
    x_training = x_data[:testing_separation_index]
    x_testing = x_data[testing_separation_index:]

    y_training = y_data[:testing_separation_index]
    y_testing = y_data[testing_separation_index:]

    # perform least squares regression
    reg = linear_model.LinearRegression()
    starting_time = time.time()
    reg.fit(x_training, y_training)
    finishing_time = time.time()
    elapsed_time = finishing_time - starting_time
    print(elapsed_time)
    print(reg.coef_)

    # predict new values
    y_predicted = reg.predict(x_testing)

    print("The root mean squared error is", mean_squared_error(y_testing, y_predicted))
    print("The r squared score is", r2_score(y_testing, y_predicted))


# still need to fix
def gradient_descent(x_data, y_data, testing_separation_index):
    # standardize x_data
    sc = StandardScaler()
    sc.fit_transform(x_data)

    # Linear regression using training data
    x_training = x_data[:testing_separation_index]
    # x_testing = x_data[testing_separation_index:]

    y_training = y_data[:testing_separation_index]
    y_testing = y_data[testing_separation_index:]

    reg = linear_model.LinearRegression()
    reg.fit(x_training, y_training)

    # perform gradient descent method
    m_current = np.random.randn(x_data.shape[1])
    b_current = 0
    iterations = 100
    n = len(x_data)
    learning_rate = 0.01

    for i in range(iterations + 1):
        # Make predictions for y = mx + b
        y_predicted = m_current * x_data + b_current

        # Calculate the cost
        cost = (1 / n) * sum([val ** 2 for val in (y_data[i] - y_predicted[i])])

        # Calculate the gradients for weight and bias
        md = -(2 / n) * sum(x_data[i] * (y_data[i] - y_predicted[i]))
        bd = -(2 / n) * sum(y_data[i] - y_predicted[i])

        # Update the current weight and bias
        m_current = m_current - learning_rate * md
        b_current = b_current - learning_rate * bd
        print("m: {}, b: {}, iteration: {}, cost: {}".format(m_current, b_current, i, cost))

    print(y_predicted.shape)  # shape should be (420000,)

    # issues with calculations due to inconsistent shapes
    # input variables with inconsistent numbers of samples: [84000, 420000]
    print("The root mean squared error is", mean_squared_error(y_testing, y_predicted))
    print("The r squared score is", r2_score(y_testing, y_predicted))


if __name__ == '__main__':
    main()

