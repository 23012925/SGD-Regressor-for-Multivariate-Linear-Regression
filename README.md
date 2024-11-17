# SGD-Regressor-for-Multivariate-Linear-Regression

## AIM:
To write a program to predict the price of the house and number of occupants in the house with SGD regressor.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required packages and print the present data. 
2. Print the placement data and salary data. 
3. Find the null and duplicate values. 
4. Using logistic regression find the predicted values of accuracy, confusion matrices. 
5. Display the results.

## Program:
```
/*
Program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor.
Developed by: JANARTHANAN K
RegisterNumber: 212223040072

  import numpy as np
  from sklearn.datasets import fetch_california_housing
  from sklearn.linear_model import SGDRegressor
  from sklearn.multioutput import MultiOutputRegressor
  from sklearn.model_selection import train_test_split
  from sklearn.metrics import mean_squared_error
  from sklearn.preprocessing import StandardScaler
  #load the california housing dataset
  data = fetch_california_housing()
  #use the first 3 features as inputs
  X= data.data[:, :3] #features: 'Medinc','housage','averooms'
  Y=np.column_stack((data.target,data.data[:, 6]))
  x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.2,random_state=42)
  #scale the features and target variables
  scaler_x = StandardScaler()
  scaler_y = StandardScaler()
  x_train = scaler_x.fit_transform(x_train)
  x_test = scaler_x.transform(x_test)
  y_train = scaler_y.fit_transform(y_train)
  y_test = scaler_y.transform(y_test)
  #initialize the SGDRegressor
  sgd = SGDRegressor(max_iter = 1000,tol = 1e-3)
  #Use Multioutputregressor to handle multiple output varibles
  multi_output_sgd = MultiOutputRegressor(sgd)
  #train the model
  multi_output_sgd.fit(x_train,y_train)
  #predict on the test data
  y_pred = multi_output_sgd.predict(x_test)
  #inverse transform the prediction to get them back to the original scale
  y_pred = scaler_y.inverse_transform(y_pred)
  y_test = scaler_y.inverse_transform(y_test)
  #evaluate the model using mean squared error
  mse = mean_squared_error(y_test,y_pred)
  print("Mean Squared Error:",mse)
  #optionally print some predictions
  print("\npredictions:\n",y_pred[:5])
*/
```

## Output:
![image](https://github.com/user-attachments/assets/ca06817d-6e9f-4691-85b9-bbffe14c55db)


## Result:
Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.
