import sys
import time
import math
import pandas as pd
import numpy as np
from sklearn import linear_model, preprocessing
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

''' develop the best predictive model based on the chemical engineering dataset'''

__author__ = 'Jacob Hajjar, Michael-Ken Okolo, Onkar Muttemwar, Vatsal Patel, Ayush Bhatnagar, '
__email__ = 'hajjarj@csu.fullerton.edu, michaelken.okolo1@csu.fullerton.edu, onkar.muttemwar@csu.fullerton.edu, vatsal1224@csu.fullerton.edu, ayush.bhatnagar@csu.fullerton.edu'
__maintainer__ = 'jacobhajjar, michaelkenokolo, onkar-muttemwar'


def main():
    """the main function"""
    data_frame = pd.read_csv("Data1.csv")
    x_data = data_frame[["T", "P", "TC", "SV"]].to_numpy()
    y_data = data_frame["Idx"].to_numpy()

    least_squares_with_libraries(x_data, y_data)
    gradient_descent(x_data, y_data)
    standard_scale(x_data, y_data)
    Fitting_Model(x_data, y_data)
    K_Fold_Func(x_data, y_data)


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
    w = np.random.randn(x_data.shape[1])
    b = 0
    learning_rate = 0.001
    num_iterations = 10000
    total_samples = x_training.shape[0]

    for i in range(num_iterations + 1):
        # Make predictions using dot product between weight(w) and x_testing
        y_predicted = w * x_testing + b

        # Calculate gradients for weight(w) and bias(b)
        w_grad = -(1 / total_samples) * (x_training[i].T.dot(y_training[i] - y_predicted[i]))
        b_grad = -(1 / total_samples) * np.sum(y_training[i] - y_predicted[i])

        # Update the current weight(w) and bias(b)
        w = w - learning_rate * w_grad
        b = b - learning_rate * b_grad

        # Calculate the cost between y_training samples and y_predicted
        cost = np.square(y_training[i] - y_predicted[i])

    print("w: {}, b: {}, iteration: {}, cost: {}".format(w, b, i, cost))

    print(y_testing, y_predicted)
    print("Root Mean Square Error: ", mean_squared_error(y_testing, y_predicted))
    print("R2: ", r2_score(y_testing, y_predicted))


def standard_scale(x_data, y_data):

# separate 80% of the data to training

    testing_separation_index = math.floor(len(x_data) * 0.8)
    x_training = x_data[:testing_separation_index]
    x_testing = x_data[testing_separation_index:]

    #Scaling the data using Standard Scaler (Data Standardization)
    scaler = StandardScaler()
    X_train_scale = scaler.fit_transform(x_training) 

    # Printing a row of data(featured) to show before and after scaling data
    print("Before Scaling: ", x_training[0])
    print("After Scaling: ", X_train_scale[0])

    y_training = y_data[:testing_separation_index]
    y_testing = y_data[testing_separation_index:]

    # perform least squares regression
    reg = linear_model.LinearRegression()
    starting_time = time.time()
    reg.fit(X_train_scale, y_training)
    finishing_time = time.time()
    elapsed_time = finishing_time - starting_time
    print("Elapsed Time: ",elapsed_time)

    # predict new values for test dataset
    X_test_scale = scaler.transform(x_testing) 
    y_predicted = reg.predict(X_test_scale)

    #predicitng target for training data
    y_train_predicted = reg.predict(X_train_scale)


    print("The test root mean squared error is", mean_squared_error(y_testing, y_predicted))
    print("The test r squared score is", r2_score(y_testing, y_predicted))

    print("The training root mean squared error is", mean_squared_error(y_training, y_train_predicted))
    print("The training r squared score is", r2_score(y_training, y_train_predicted))


def K_Fold_Func(X, y):

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)

#def Ridge_Model():
    ridge = Ridge(alpha = 1.0)
    mymodel = ridge.fit(X_test, y_test)
    RidgeCV = cross_val_score(mymodel, X_test, y_test, scoring = 'r2', cv = 10)
    print('K-fold CV (Ridge) -',RidgeCV)
    print('Mean -',np.mean(RidgeCV))

#def Lasso_Model():    
    lasso =  Lasso(alpha =0.0001)
    mymodel1 = lasso.fit(X_test, y_test)
    LassoCV = cross_val_score(mymodel1, X_test, y_test, scoring = 'r2', cv = 10)
    print('K-fold CV (Lasso) -',LassoCV)
    print('Mean -',np.mean(LassoCV))

#def ElasticNet_Model():    
    elasticnet = ElasticNet(alpha = 0.01)
    mymodel2 = elasticnet.fit(X_test, y_test)
    ElasticNetCV = cross_val_score(mymodel2, X_test, y_test, scoring = 'r2', cv = 10)
    print('K-fold CV (Elastic-Net) -',ElasticNetCV)
    print('Mean -',np.mean(ElasticNetCV))


def Fitting_Model(x,y):
    
    global X_train
    global X_test
    global y_train
    global y_test
    X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.3)

    print (len(X_test), len(y_test))

    lr = LinearRegression()
    lr.fit(X_train, y_train)
    Y_train_new = lr.predict(X_train)
    Y_test_new = lr.predict(X_test)
    print(f'Linear Regression score {lr.score(X_train,y_train)} Linear Regression test score {lr.score(X_test,y_test)} ')
    rmse = mean_squared_error(y_train, Y_train_new , squared=False)
    print(f'RMSE for linear regression {rmse}\n\n')
    # rmse = math.sqrt(mse)


    start=time.time()
    # lasso_reg= Lasso(alpha=0.0004)
    lasso_reg = Pipeline(steps=[
    ('scaler',StandardScaler()),
    ('Lasso',Lasso(alpha=0.01,random_state=1000))
    ])
    lasso_reg.fit(X_train,y_train)
    end=time.time()
    # lasso_reg.fit(X_test,y_test)
    print(f"Time taken by Lasso model to learned and train {end-start}")
    lasso_reg_test_new= lasso_reg.predict(X_test)
    lasso_reg_train_new= lasso_reg.predict(X_train)
    lasso_reg_test_rmse=mean_squared_error(y_test,lasso_reg_test_new,squared=False)
    lasso_reg_train_rmse=mean_squared_error(y_train,lasso_reg_train_new,squared=False)
    lasso_reg_train_r2 = r2_score(y_train,lasso_reg_train_new)
    lasso_reg_test_r2 = r2_score(y_test,lasso_reg_test_new)
    print(f"Lasso_reg_train_RMSE: {lasso_reg_test_rmse}\nLasso_reg_test_RMSE: {lasso_reg_train_rmse}\nLasso test R2 Score: {lasso_reg_test_r2}\nLasso train R2 score {lasso_reg_train_r2}\n \n")


    start_ridge_time = time.time()
    # rr = Ridge(alpha=0.05, normalize=True)
    # Instantiate a lasso regressor: lasso
    #lasso = Lasso(alpha=0.4, normalize=True)
    rr = Pipeline(steps=[
    ('scaler',StandardScaler()),
    ('ridge',Ridge(alpha=0.01,random_state=1000))
    ])
    # higher the alpha value, more restriction on the coefficients; low alpha > more generalization,
    # in this case linear and ridge regression resembles
    rr.fit(X_train, y_train)
    end_ridge_time = time.time()
    print(f'Time taken by ridge model to learned and train {end_ridge_time-start_ridge_time}')
    rr_test_new= rr.predict(X_test)
    rr_train_new= rr.predict(X_train)
    rr_test_rmse = mean_squared_error(y_test, rr_test_new, squared=False)
    rr_train_rmse = mean_squared_error(y_train, rr_train_new, squared=False)
    print(f"ridge_test_RMSE: {rr_test_rmse}")
    print(f"ridge_train_RMSE: {rr_train_rmse}")
    # train_score=lr.score(X_train, y_train)
    # test_score=lr.score(X_test, y_test)
    Ridge_test_r2_score = r2_score(y_test,rr_test_new)
    Ridge_train_r2_score = r2_score(y_train,rr_train_new)
    print(f'Ridge test R2 Score {Ridge_test_r2_score}\nRidge train R2 Score {Ridge_train_r2_score}\n\n')



    start_elastic_time = time.time()
    # elastic_net = ElasticNet(alpha=0.01, random_state=1000)
    elastic_net = Pipeline(steps=[
    ('scaler',StandardScaler()),
    ('ridge',ElasticNet(alpha=0.01,random_state=1000))
    ])
    elastic_net.fit(X_train, y_train)
    end_elastic_time = time.time()
    print(f'Time taken by elastic_net to learned and train {end_elastic_time-start_elastic_time}')
    # calculate the prediction and mean square error
    y_pred_elastic_test = elastic_net.predict(X_test)
    y_pred_elastic_train = elastic_net.predict(X_train)
    # mean_squared_error = np.mean((y_pred_elastic - y_test) ** 2)
    elastic_net_test_R2_score= r2_score(y_test,y_pred_elastic_test)
    elastic_net_train_R2_score= r2_score(y_train,y_pred_elastic_train)
    elastic_net_test_rmse = mean_squared_error(y_test, y_pred_elastic_test, squared=False)
    elastic_net_train_rmse = mean_squared_error(y_train, y_pred_elastic_train, squared=False)
    print(f"elastic net test R2 score{elastic_net_test_R2_score}\nelastic net train R2 score{elastic_net_train_R2_score}\nelastic net test RMSE {elastic_net_test_rmse}\nelastic net train RMSE {elastic_net_train_rmse}")



# plt.plot(rr.coef_,alpha=0.7,linestyle='none',marker='*',markersize=5,color='red',label=r'Ridge; $\alpha = 1$',zorder=7)
# plt.plot(lasso_reg.coef_,alpha=0.5,linestyle='none',marker='d',markersize=6,color='blue',label=r'Lasso; $\alpha = 0.005$')
# plt.plot(lr.coef_,alpha=0.4,linestyle='none',marker='o',markersize=7,color='green',label='Linear Regression')
# plt.xlabel('Coefficient Index',fontsize=16)
# plt.ylabel('Coefficient Magnitude',fontsize=16)
# plt.legend(fontsize=13,loc=4)
# plt.show()

if __name__ == '__main__':
    main()
