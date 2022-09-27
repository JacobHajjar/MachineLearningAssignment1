Objective: Linear Regression Modeling 
--------------------------------------------------------------------------------------------------------------------- 
Background and problem description: 
Chemical engineers measured various properties of a gas and created a training data set with four features temperature (T), pressure (P), thermal conductivity (TC), and sound velocity (SV). They also calculated the gas quality using four measured properties and converted it into a quality index (Idx). They want to know if any functional relationships exist between the measured properties and the gas quality. The schema of the data set is:
Dataset(T, P, TC, SV, Idx) where the type of each attribute is double.

This program develops the best predictive model for the data1.csv training data set using Linear Regression, Gradient Descent, Feature Scaling, LASSO, and K-Fold Cross Validation.
