#!/usr/bin/env python3
import sys
from telnetlib import theNULL
import time
import math
import pandas as pd
import numpy as np
from sklearn import linear_model, preprocessing
from sklearn.metrics import mean_squared_error, r2_score

''' develop the best predictive model based on the chemical engineering dataset'''

__author__ = 'Jacob Hajjar, Michael-Ken Okolo'
__email__ = 'hajjarj@csu.fullerton.edu, michaelken.okolo1@csu.fullerton.edu'
__maintainer__ = 'jacobhajjar, michaelkenokolo'


def main():
    """the main function"""
    data_frame = pd.read_csv("Data1.csv")
    x_data = data_frame[["T", "P", "TC", "SV"]].to_numpy()
    y_data = data_frame["Idx"].to_numpy()

    #least_squares_with_libraries(x_data, y_data)
    gradient_descent(x_data, y_data)


def least_squares_with_libraries(x_data, y_data):
    # separate 80% of the data to training
    testing_separation_index = math.floor(len(x_data) * 0.8)
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
    y_predicted_test = reg.predict(x_testing)
    y_predicted_training = reg.predict(x_training)

    print("The root mean squared error for the testing data is", mean_squared_error(y_testing, y_predicted_test))
    print("The r squared score for the testing data is", r2_score(y_testing, y_predicted_test))

    print("The root mean squared error for the training data is", mean_squared_error(y_training, y_predicted_training))
    print("The r squared score for the training data is", r2_score(y_training, y_predicted_training))


# still need to fix
def gradient_descent(x_data, y_data):
    # separate 80% of the data to training
    testing_separation_index = math.floor(len(x_data) * 0.8)
    x_training = x_data[:testing_separation_index]
    x_testing = x_data[testing_separation_index:]

    y_training = y_data[:testing_separation_index]
    y_testing = y_data[testing_separation_index:]

    # perform gradient descent method
    w = np.random.randn(x_data.shape[1] + 1)
    learning_rate = 0.001
    num_iterations = 10000
    total_samples = x_training.shape[0]

    #for i in range(num_iterations + 1):
    converged = False
    prev_weight_change = [0, 0, 0, 0, 0]
    while not converged:
        for j_index, wj in enumerate(w):
            sum_of_difference = 0
            for i_index, x in enumerate(x_training):
                padded_x = np.append(x, [1])
                sum_of_difference += (((np.dot(w, padded_x) - y_training[i_index])) * padded_x[j_index])
            weight_change = (learning_rate / total_samples) * sum_of_difference
            w[j_index] = wj - weight_change

            if prev_weight_change[j_index] < 0 and weight_change > 0:
                converged = True
            else:
                prev_weight_change[j_index] = weight_change
    #print(w)

    #print("w: {}, b: {}, iteration: {}, cost: {}".format(w, b, i, cost))

    #print(y_testing, y_predicted)
    #print("Root Mean Square Error: ", mean_squared_error(y_testing, y_predicted))
    #print("R2: ", r2_score(y_testing, y_predicted))


if __name__ == '__main__':
    main()
