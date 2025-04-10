# Task B: Wind Power Prediction using Linear Regression
# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('Data/TrainData.csv')

# Calculate wind direction
# Wind direction in degrees: 0° is north, 90° is east, 180° is south, 270° is west
data['wind_direction'] = (np.arctan2(data['V10'], data['U10']) * (180 / np.pi) + 360) % 360

# Separate the features (X) and the target variable (y)
X = data[['wind_direction', 'WS10']]  # Use wind_direction and WS10 as input features
y = data['POWER']  # Use POWER as the target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R-squared:", r2)

# Load the forecasted wind data
forecast_data = pd.read_csv('Data/WeatherForecastInput.csv')
forecast_template = pd.read_csv('Data/ForecastTemplate.csv')

# Calculate wind direction for the forecasted data
forecast_data['wind_direction'] = (np.arctan2(forecast_data['V10'], forecast_data['U10']) * (180 / np.pi) + 360) % 360

# Prepare the input features for prediction
forecast_X = forecast_data[['wind_direction', 'WS10']]

# Predict the power output using the trained model
forecast_template['FORECAST'] = model.predict(forecast_X)

# Save the predictions to a new CSV file
forecast_template.to_csv('Results/ForecastTemplate2.csv', index=False)

print("Predictions saved to 'Results/ForecastTemplate2.csv'")

# Load the actual values from Solution.csv
solution_data = pd.read_csv('Data/Solution.csv')

# Ensure the predicted and actual data align
if 'FORECAST' in forecast_template.columns and 'POWER' in solution_data.columns:
    # Calculate RMSE
    rmse = np.sqrt(mean_squared_error(solution_data['POWER'], forecast_template['FORECAST']))
    print("Root Mean Squared Error (RMSE):", rmse)
else:
    print("Error: Required columns are missing in the data.")




