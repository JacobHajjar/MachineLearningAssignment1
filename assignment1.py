#!/usr/bin/env python3
import sys
import time
import math
import pandas as pd
import numpy as np
from sklearn import linear_model
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

    least_squares_with_libraries(x_data, y_data)
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
    starting_time = time.time()

    w = np.random.randn(x_data.shape[1])
    b = 0
    learning_rate = 0.01
    num_iterations = 1000
    total_samples = x_training.shape[0]

    for i in range(num_iterations + 1):
        # Make predictions using dot product between weight(w) and x_testing
        y_predicted = np.dot(w, x_testing.T) + b

        # Calculate gradients for weight(w) and bias(b)
        w_grad = -(2 / total_samples) * (x_training[i].T.dot(y_training[i] - y_predicted[i]))
        b_grad = -(2 / total_samples) * np.sum(y_training[i] - y_predicted[i])

        # Update the current weight(w) and bias(b)
        w = w - learning_rate * w_grad
        b = b - learning_rate * b_grad

        # Calculate the cost between y_training samples and y_predicted
        cost = np.square(y_training[i] - y_predicted[i])

    print("w: {}, b: {}, iteration: {}, cost: {}".format(w, b, i, cost))

    finishing_time = time.time()
    elapsed_time = finishing_time - starting_time
    print(elapsed_time)

    print("Root Mean Square Error: ", mean_squared_error(y_testing, y_predicted))
    print("R2: ", r2_score(y_testing, y_predicted))


if __name__ == '__main__':
    main()
