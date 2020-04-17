
# # importing modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (12.0, 9.0)

# reading in data from the csv file
data = pd.read_csv('data.csv')
# stores all the input data in a n * 1 dimensional vector
X = data.iloc[:, 0]
# stores all the expected outputs in a n * 1 dimensional vector
Y = data.iloc[:, 1]
# plots raw data
plt.scatter(X, Y)
plt.show()

# initialise theta values to 0
theta_1 = 0
theta_0 = 0

alpha = 0.1  # learning Rate
iters = 1000  # number of iterations to perform gradient descent

m = float(len(X)) # number of training examples

# gradient descent algorithm
for i in range(iters): 
    Y_pred = theta_1 * X + theta_0  # The current predicted value of Y
    D_1 = (1 / m) * sum((Y_pred - Y) * X)  # Derivative wrt m
    D_0 = (1 / m) * sum(Y_pred - Y)  # Derivative wrt c
    theta_1 = theta_1 - alpha * D_1  # Update m
    theta_0 = theta_0 - alpha * D_0  # Update c

# computes values for theta 1 and theta 0
print(theta_1, theta_0)

# plotting the line of best fit (prediction)
Y_pred = theta_1 * X + theta_0

plt.scatter(X, Y) # raw data
plt.plot([min(X), max(X)], [min(Y_pred), max(Y_pred)], color='red')  # regression line
plt.show()