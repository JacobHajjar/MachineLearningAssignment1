#!/usr/bin/env python3
import sys
import time
import math
import pandas as pd
import numpy as np
from numpy import random
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

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
    stochastic_gradient_descent(x_data, y_data, testing_separation_index)


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
def stochastic_gradient_descent(x_data, y_data, testing_separation_index):
    # separate 80% of the data to training
    x_training = x_data[:testing_separation_index]
    x_testing = x_data[testing_separation_index:]

    y_training = y_data[:testing_separation_index]
    y_testing = y_data[testing_separation_index:]

    number_of_features = x_training.shape[1]  # 4 features

    # perform gradient descent method
    w = np.ones(shape=number_of_features)  # shape [1. 1. 1. 1.]
    b = 0
    total_samples = x_training.shape[0]  # N = total_samples
    epochs = 100000  # number of iterations
    learning_rate = 0.5

    cost_list = []
    epoch_list = []

    for i in range(epochs + 1):
        # separate 80% of the data to training
        random_index = random.randint(0, total_samples - 1)
        sample_x = x_training[random_index]
        sample_y = y_training[random_index]

        # Make predictions for y = mx + b
        y_predicted = np.dot(w, sample_x.T) + b

        # Calculate the gradients for weight(w) and bias(b)
        m_grad = -(2 / total_samples) * (sample_x.T.dot(sample_y - y_predicted))
        b_grad = -(2 / total_samples) * np.sum(sample_y - y_predicted)

        # Update the current weight(w) and bias(b)
        w = w - learning_rate * m_grad
        b = b - learning_rate * b_grad

        # Calculate the cost
        cost = np.square(sample_y - y_predicted)

        if i % 1000 == 0:
            cost_list.append(cost)
            epoch_list.append(i)

    # unsure if this value is correct, but set the iterations to 100,000 and got this.
    print("m: {}, b: {}, epoch: {}, cost: {}".format(w, b, i, cost))
    # print("The root mean squared error is", mean_squared_error(y_testing, y_predicted))
    # print("The r squared score is", r2_score(y_training, y_predicted))


if __name__ == '__main__':
    main()
