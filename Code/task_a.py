# Task A: Wind Power Prediction using Multiple Models
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
import matplotlib.dates as mdates
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, loguniform

# Load the dataset
data = pd.read_csv('Data/TrainData.csv')

# Separate the features (X) and the target variable (y)
X = data['WS10']  # Use WS10 as input features
y = data['POWER']  # Use POWER as the target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
X_train = X_train.values.reshape(-1, 1)
X_test = X_test.values.reshape(-1, 1)

# Create and train the linear regression model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Train a KNN model
knn_model = KNeighborsRegressor()

knn_param_distributions = {
    'n_neighbors': range(2,1000),  # Number of neighbors to use
    'weights': ['uniform', 'distance'],
    'p': [1, 2]  # 1 = Manhattan distance, 2 = Euclidean distance
}

knn_random_search = RandomizedSearchCV(
    estimator=knn_model,                  # use svr, ridge, knn, or mlp
    param_distributions=knn_param_distributions,        # the corresponding params
    n_iter=50,
    scoring='neg_mean_squared_error',
    cv=5,
    verbose=2,
    random_state=42,
    n_jobs=-1
)

knn_random_search.fit(X_train, y_train)
# Best parameters
print("(KNN) Best parameters found:", knn_random_search.best_params_)

# Train an SVR model
svr_model = SVR(kernel='poly')

# Perform grid search for SVR model
# Define the hyperparameter space
svr_param_distributions = {
    'C': loguniform(1e-2, 1e3),
    'epsilon': uniform(0, 0.5),
    #'kernel': ['rbf', 'poly'],
    'gamma': ['scale', 'auto', loguniform(1e-4, 1e-1)],  # important for both
    'degree': [2, 3, 4, 5]  # only used if kernel='poly'            
}

# Set up RandomizedSearchCV
svr_random_search = RandomizedSearchCV(
    estimator=svr_model,
    param_distributions=svr_param_distributions,
    n_iter=50,                # Number of parameter combinations to try
    scoring='neg_mean_squared_error',
    cv=5,                     # 5-fold cross-validation
    verbose=2,
    random_state=42,
    n_jobs=-1
)

# Fit on your training data
svr_random_search.fit(X_train, y_train)

# Best parameters
print("(SVR) Best parameters found:", svr_random_search.best_params_)

# Train an NN model
nn_model = MLPRegressor(activation='relu', max_iter=5000)

nn_param_distributions = {
    'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50), (100, 100)],
    'solver': ['adam', 'lbfgs'],
    'alpha': loguniform(1e-5, 1e-2),  # regularization
    'learning_rate': ['constant', 'adaptive']
}

nn_random_search = RandomizedSearchCV(
    estimator=nn_model,                  # use svr, ridge, knn, or mlp
    param_distributions=nn_param_distributions,        # the corresponding params
    n_iter=50,
    scoring='neg_mean_squared_error',
    cv=5,
    verbose=2,
    random_state=42,
    n_jobs=-1
)

nn_random_search.fit(X_train, y_train)
# Best parameters
print("(NN) Best parameters found:", nn_random_search.best_params_)

# Make predictions for all models
lr_pred = lr_model.predict(X_test)
knn_pred = knn_random_search.predict(X_test)
svr_pred = svr_random_search.predict(X_test)
nn_pred = nn_random_search.predict(X_test)

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
forecast_X = forecast_data['WS10'].values.reshape(-1, 1)

# Predict the power output using the trained model
ln_forecast_template['FORECAST'] = lr_model.predict(forecast_X)
ln_forecast_template.to_csv('Results/ForecastTemplate1-LR.csv', index=False)
# Predict the power output using the trained model
knn_forecast_template['FORECAST'] = knn_random_search.predict(forecast_X)
knn_forecast_template.to_csv('Results/ForecastTemplate1-kNN.csv', index=False)
# Predict the power output using the trained model
svr_forecast_template['FORECAST'] = svr_random_search.predict(forecast_X)
svr_forecast_template.to_csv('Results/ForecastTemplate1-SVR.csv', index=False)
# Predict the power output using the trained model
nn_forecast_template['FORECAST'] = nn_random_search.predict(forecast_X)
nn_forecast_template.to_csv('Results/ForecastTemplate1-NN.csv', index=False)

# Load the actual values from Solution.csv
solution_data = pd.read_csv('Data/Solution.csv')

# Calculate LN_RMSE
ln_rmse = np.sqrt(mean_squared_error(solution_data['POWER'], ln_forecast_template['FORECAST']))
print("LN Root Mean Squared Error (RMSE):", ln_rmse)
# Calculate kNN RMSE
knn_rmse = np.sqrt(mean_squared_error(solution_data['POWER'], knn_forecast_template['FORECAST']))
print("kNN Root Mean Squared Error (RMSE):", knn_rmse)
# Calculate svr_RMSE
svr_rmse = np.sqrt(mean_squared_error(solution_data['POWER'], svr_forecast_template['FORECAST']))
print("SVR Root Mean Squared Error (RMSE):", svr_rmse)
# Calculate NN_RMSE
nn_rmse = np.sqrt(mean_squared_error(solution_data['POWER'], nn_forecast_template['FORECAST']))
print("NN Root Mean Squared Error (RMSE):", nn_rmse)

# Plot the POWER value vs FORECAST value for each model
# Ensure TIMESTAMP is in datetime format
solution_data['TIMESTAMP'] = pd.to_datetime(solution_data['TIMESTAMP'], format='%Y%m%d %H:%M')

# Generate a range of all days in the month
all_days = pd.date_range(start=solution_data['TIMESTAMP'].min().date(), 
                         end=solution_data['TIMESTAMP'].max().date(), freq='D')

# Plot Linear Regression
plt.figure(figsize=(15, 6))
plt.plot(solution_data['TIMESTAMP'], solution_data['POWER'], label='Actual POWER', color='blue')
plt.plot(solution_data['TIMESTAMP'], ln_forecast_template['FORECAST'], label='LR FORECAST', color='red')
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d'))  # Format x-axis to show day
plt.gca().xaxis.set_major_locator(mdates.DayLocator())  # Show all days
plt.xticks(all_days, [day.strftime('%d') for day in all_days], rotation=45)
plt.xlabel('Day')
plt.ylabel('Power')
plt.title('Linear Regression: Actual vs Forecast')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

# Plot KNN
plt.figure(figsize=(15, 6))
plt.plot(solution_data['TIMESTAMP'], solution_data['POWER'], label='Actual POWER', color='blue')
plt.plot(solution_data['TIMESTAMP'], knn_forecast_template['FORECAST'], label='KNN FORECAST', color='green')
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d'))  # Format x-axis to show day
plt.gca().xaxis.set_major_locator(mdates.DayLocator())  # Show all days
plt.xticks(all_days, [day.strftime('%d') for day in all_days], rotation=45)
plt.xlabel('Day')
plt.ylabel('Power')
plt.title('KNN: Actual vs Forecast')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

# Plot SVR
plt.figure(figsize=(15, 6))
plt.plot(solution_data['TIMESTAMP'], solution_data['POWER'], label='Actual POWER', color='blue')
plt.plot(solution_data['TIMESTAMP'], svr_forecast_template['FORECAST'], label='SVR FORECAST', color='orange')
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d'))  # Format x-axis to show day
plt.gca().xaxis.set_major_locator(mdates.DayLocator())  # Show all days
plt.xticks(all_days, [day.strftime('%d') for day in all_days], rotation=45)
plt.xlabel('Day')
plt.ylabel('Power')
plt.title('SVR: Actual vs Forecast')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

# Plot Neural Network
plt.figure(figsize=(15, 6))
plt.plot(solution_data['TIMESTAMP'], solution_data['POWER'], label='Actual POWER', color='blue')
plt.plot(solution_data['TIMESTAMP'], nn_forecast_template['FORECAST'], label='NN FORECAST', color='purple')
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d'))  # Format x-axis to show day
plt.gca().xaxis.set_major_locator(mdates.DayLocator())  # Show all days
plt.xticks(all_days, [day.strftime('%d') for day in all_days], rotation=45)
plt.xlabel('Day')
plt.ylabel('Power')
plt.title('Neural Network: Actual vs Forecast')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()
