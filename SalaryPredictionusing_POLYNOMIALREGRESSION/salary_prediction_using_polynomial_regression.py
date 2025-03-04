# -*- coding: utf-8 -*-
"""Salary prediction using POLYNOMIAL REGRESSION.ipynb

# **Salary prediction using POLYNOMIAL REGRESSION**

### *Importing Libraries*
"""

import pandas as pd

"""### *Load Dataset from Local directory*"""

from google.colab import files
uploaded = files.upload()

"""### *Load Dataset*"""

dataset = pd.read_csv('dataset.csv')

"""### *Summarize Dataset*"""

print(dataset.shape)
print(dataset.head(5))

"""### *Segregate Dataset into Input X & Output Y*"""

X = dataset.iloc[:, :-1].values
X

Y = dataset.iloc[:, -1].values
Y

"""### *Training Dataset using Linear Regression*"""

from sklearn.linear_model import LinearRegression
modelLR = LinearRegression()
modelLR.fit(X,Y)

"""### *Visualizing Linear Regression results*"""

import matplotlib.pyplot as plt
plt.scatter(X,Y, color="red")
plt.plot(X, modelLR.predict(X))
plt.title("Linear Regression")
plt.xlabel("Level")
plt.ylabel("Salary")
plt.show()

"""### *fit() - Training Model - Calculating the initial parameters*

### *transform() - After Training we gonna transform Data by using above calculated values*

### *fit_transform() - First fit & Transform*

###*Convert X to Polynomial Format (X^n)*
###*n-degree*
###*n=2 consist x & x^2*
###*n=3 consist x & x^2 & x^3*
"""

from sklearn.preprocessing import PolynomialFeatures
modelPR = PolynomialFeatures(degree = 4)
xPoly = modelPR.fit_transform(X)

"""###*Train same Linear Regression with X-Polynomial instead of X*"""

modelPLR = LinearRegression()
modelPLR.fit(xPoly,Y)

"""### *Visualizing Polynomial Regression results*"""

plt.scatter(X,Y, color="red")
plt.plot(X, modelPLR.predict(modelPR.fit_transform(X)))
plt.title("Polynomial Regression")
plt.xlabel("Level")
plt.ylabel("Salary")
plt.show()

"""### *Prediction using Polynomial Regression*"""

x=5
salaryPred = modelPLR.predict(modelPR.fit_transform([[x]]))
print('Salary of a person with Level {0} is {1}'.format(x,salaryPred))
