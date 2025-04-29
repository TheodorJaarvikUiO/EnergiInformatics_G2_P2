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

# X_train and y_train are the input features and target variable respectively
X_train = data[['wind_direction', 'WS10']]  # Use wind_direction and WS10 as input features
y_train = data['POWER']  # Use POWER as the target variable

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

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

# Evaluate the model
rmse = np.sqrt(mean_squared_error(solution_data['POWER'], forecast_template['FORECAST']))
r2 = r2_score(solution_data['POWER'], forecast_template['FORECAST'])

print("Multiple LR RMSE:", rmse)
print("Multiple LR R-squared:", r2)

### SIMPLE LINEAR REGRESSION ###
X_train_lr = data['WS10']
y_train_lr = data['POWER']

# Create and train the linear regression model
lr_model = LinearRegression()
lr_model.fit(X_train_lr.values.reshape(-1, 1), y_train_lr)

forecast_X_lr = forecast_data['WS10']
forecast_template_lr = pd.read_csv('Data/ForecastTemplate.csv')
# Predict the power output using the trained model
forecast_template_lr['FORECAST'] = lr_model.predict(forecast_X_lr.values.reshape(-1, 1))

# Evaluate the model
lr_rmse = np.sqrt(mean_squared_error(solution_data['POWER'], forecast_template_lr['FORECAST']))
lr_r2 = r2_score(solution_data['POWER'], forecast_template_lr['FORECAST'])

print("SIMPLE LR RMSE:", lr_rmse)
print("SIMPLE LR R-squared:", lr_r2)

# Convert TIMESTAMP to datetime for proper plotting
solution_data['TIMESTAMP'] = pd.to_datetime(solution_data['TIMESTAMP'], format='%Y%m%d %H:%M')

# Generate a range of all days in the month
all_days = pd.date_range(start=solution_data['TIMESTAMP'].min().date(), 
                         end=solution_data['TIMESTAMP'].max().date(), freq='D')
# Plot the data
plt.figure(figsize=(15, 6))
plt.plot(solution_data['TIMESTAMP'], solution_data['POWER'], label='Actual Power (Solution)', color='blue')
plt.plot(solution_data['TIMESTAMP'], forecast_template['FORECAST'], label='Forecasted Power (Multiple LR Model)', color='orange')
plt.plot(solution_data['TIMESTAMP'], forecast_template_lr['FORECAST'], label='Forecasted Power (Simple LR Model)', color='green')
plt.xticks(all_days, [day.strftime('%d') for day in all_days], rotation=45)

# Add labels, legend, and title
plt.xlabel('Date')
plt.ylabel('Power')
plt.title('Power Comparison: Actual vs Forecasted')
plt.legend()
plt.grid()
plt.tight_layout()

# Show the plot
plt.show()