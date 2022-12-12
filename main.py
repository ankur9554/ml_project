#Importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Importing Data Set

dataset = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/Dataset/RBC in Humans Polynomial Regres.csv')
X= dataset.iloc[:,0:-1].values
y= dataset.iloc[:,-1].values

#Training the Linear Regression model on the whole dataset

from sklearn.linear_model import LinearRegression
linear_regression = LinearRegression()
linear_regression.fit(X,y)
LinearRegression()

#Training the Polynomial Regression model on the whole dataset

from sklearn.preprocessing import PolynomialFeatures
polynomial_reg = PolynomialFeatures(degree = 8)
X_poly = polynomial_reg.fit_transform(X)
linear_reg2 = LinearRegression()
linear_reg2.fit(X_poly,y)
LinearRegression()

#Visualising the Linear Regression results

plt.scatter(X,y, color= 'red')
plt.plot(X, linear_regression.predict(X))
plt.title('Age vs RBC Linear')
plt.xlabel('Age')
plt.ylabel('RBC')
plt.show()

#Visualising the Polynomial Regression results (for higher resolution and smoother curve)

plt.scatter(X,y, color= 'red')
plt.plot(X, linear_reg2.predict(X_poly))
plt.title('Age vs RBC Polynomail')
plt.xlabel('Age')
plt.ylabel('RBC')
plt.show()

#Predicting a new result with Linear Regression

linear_regression.predict([[6.5]])
#array([1980.15863901])

#Predicting a new result with Polynomial Regression

linear_reg2.predict(polynomial_reg.fit_transform([[6.5]]))
#array([799.26093977])
