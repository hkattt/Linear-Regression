
# # importing modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# setting the figures size
plt.rcParams['figure.figsize'] = (12.0, 9.0)

class LinearRegression():
    def __init__(self):
        # reading in data from the csv file
        self.data = pd.read_csv("data_2.csv")
        # stores all the input data in a n * 1 dimensional vector
        self.X = self.data.iloc[:, 0]
        # stores all the expected outputs in a n * 1 dimensional vector
        self.Y = self.data.iloc[:, 1]
        # initialise theta values to 0
        self.theta_1 = 0
        self.theta_0 = 0

        self.alpha = 0.01  # learning Rate
        self.iters = 1000  # number of iterations to perform gradient descent

        self.m = float(len(self.X)) # number of training examples

    def plot_raw(self):
        """ Plots raw data """
        plt.scatter(self.X, self.Y) # plots all the data points
        # axis labels / title
        plt.title("Years of Experience vs Salary")
        plt.xlabel("Years of Experience")
        plt.ylabel("Salary ($)")
        # displays figure
        plt.show()

    def gradient_descent(self):
        """ Performs gradient descent """
        # iterates over all the training iterations (epochs)
        for _ in range(self.iters): 
            h = self.theta_1 * self.X + self.theta_0  # hypothesis
            D_1 = (1 / self.m) * sum((h - self.Y) * self.X)  # partial derivative with respect to theta 1
            D_0 = (1 / self.m) * sum(h - self.Y)  # partial derivative with respect to theta 0
            self.theta_1 = self.theta_1 - self.alpha * D_1  # update theta 1
            self.theta_0 = self.theta_0 - self.alpha * D_0  # update theta 0

    def plot(self):
        """ Plots the raw data with the regression line """
        h = self.theta_1 * self.X + self.theta_0 # hypothesis function
        plt.scatter(self.X, self.Y) # raw data
        plt.plot([min(self.X), max(self.X)], [min(h), max(h)], label="Regression Line", color="red")  # regression line (hypothesis function)
        # axis labels / title
        plt.title("Years of Experience vs Salary")
        plt.xlabel("Years of Experience")
        plt.ylabel("Salary ($)")
        plt.legend()
        # displays figure
        plt.show()

linear_regression = LinearRegression()
linear_regression.plot_raw()
linear_regression.gradient_descent()
print(linear_regression.theta_1, linear_regression.theta_0)
linear_regression.plot()