# Task B: Wind Power Prediction using Linear Regression
# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('Data/TrainData.csv')

# Separate the features (X) and the target variable (y)
X = data['WS10']  # Use WS10 as input features
y = data['POWER']  # Use POWER as the target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Create and train the linear regression model
lr_model = LinearRegression()
lr_model.fit(X_train.values.reshape(-1, 1), y_train)

# Train a KNN model
knn_model = KNeighborsRegressor(n_neighbors=5)
knn_model.fit(X_train.values.reshape(-1, 1), y_train)

# Train an SVR model
svr_model = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1)
svr_model.fit(X_train.values.reshape(-1, 1), y_train)

# Train an NN model
nn_model = MLPRegressor(hidden_layer_sizes=(100,), activation='relu', solver='adam', max_iter=500, random_state=42)
nn_model.fit(X_train.values.reshape(-1, 1), y_train)

# Make predictions for all models
lr_pred = lr_model.predict(X_test.values.reshape(-1, 1))
knn_pred = knn_model.predict(X_test.values.reshape(-1, 1))
svr_pred = svr_model.predict(X_test.values.reshape(-1, 1))
nn_pred = nn_model.predict(X_test.values.reshape(-1, 1))

# Evaluate all models
print("LR - MSE:", mean_squared_error(y_test, lr_pred), "R2:", r2_score(y_test, lr_pred))
print("KNN - MSE:", mean_squared_error(y_test, knn_pred), "R2:", r2_score(y_test, knn_pred))
print("SVR - MSE:", mean_squared_error(y_test, svr_pred), "R2:", r2_score(y_test, svr_pred))
print("NN - MSE:", mean_squared_error(y_test, nn_pred), "R2:", r2_score(y_test, nn_pred))

# Load the forecasted wind data
forecast_data = pd.read_csv('Data/WeatherForecastInput.csv')
ln_forecast_template = pd.read_csv('Data/ForecastTemplate.csv')
knn_forecast_template = pd.read_csv('Data/ForecastTemplate.csv')
svr_forecast_template = pd.read_csv('Data/ForecastTemplate.csv')
nn_forecast_template = pd.read_csv('Data/ForecastTemplate.csv')

# Prepare the input features for prediction
forecast_X = forecast_data['WS10']

# Predict the power output using the trained model
ln_forecast_template['FORECAST'] = lr_model.predict(forecast_X.values.reshape(-1, 1))
ln_forecast_template.to_csv('Results/ForecastTemplate1-LR.csv', index=False)
# Predict the power output using the trained model
knn_forecast_template['FORECAST'] = knn_model.predict(forecast_X.values.reshape(-1, 1))
knn_forecast_template.to_csv('Results/ForecastTemplate1-kNN.csv', index=False)
# Predict the power output using the trained model
svr_forecast_template['FORECAST'] = svr_model.predict(forecast_X.values.reshape(-1, 1))
svr_forecast_template.to_csv('Results/ForecastTemplate1-SVR.csv', index=False)
# Predict the power output using the trained model
nn_forecast_template['FORECAST'] = nn_model.predict(forecast_X.values.reshape(-1, 1))
nn_forecast_template.to_csv('Results/ForecastTemplate1-NN.csv', index=False)

# Load the actual values from Solution.csv
solution_data = pd.read_csv('Data/Solution.csv')

# Ensure the predicted and actual data align
if 'FORECAST' in ln_forecast_template.columns and 'POWER' in solution_data.columns:
    # Calculate RMSE
    rmse = np.sqrt(mean_squared_error(solution_data['POWER'], ln_forecast_template['FORECAST']))
    print("LN Root Mean Squared Error (RMSE):", rmse)
else:
    print("Error: Required columns are missing in the data.")
   
    # Ensure the predicted and actual data align
if 'FORECAST' in knn_forecast_template.columns and 'POWER' in solution_data.columns:
    # Calculate RMSE
    rmse = np.sqrt(mean_squared_error(solution_data['POWER'], knn_forecast_template['FORECAST']))
    print("kNN Root Mean Squared Error (RMSE):", rmse)
else:
    print("Error: Required columns are missing in the data.")
   
    # Ensure the predicted and actual data align
if 'FORECAST' in svr_forecast_template.columns and 'POWER' in solution_data.columns:
    # Calculate RMSE
    rmse = np.sqrt(mean_squared_error(solution_data['POWER'], svr_forecast_template['FORECAST']))
    print("SVR Root Mean Squared Error (RMSE):", rmse)
else:
    print("Error: Required columns are missing in the data.")
   
    # Ensure the predicted and actual data align
if 'FORECAST' in nn_forecast_template.columns and 'POWER' in solution_data.columns:
    # Calculate RMSE
    rmse = np.sqrt(mean_squared_error(solution_data['POWER'], nn_forecast_template['FORECAST']))
    print("NN Root Mean Squared Error (RMSE):", rmse)
else:
    print("Error: Required columns are missing in the data.")
