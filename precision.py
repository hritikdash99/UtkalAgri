import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Assuming you have a dataset containing historical data on soil moisture, nutrient levels, crop health, and weather conditions
agricultural_data = pd.read_csv('Semicolon ragi.csv')

# Preprocess the data
# Example: Extract relevant features such as soil moisture, nutrient levels, crop health, and weather conditions
X = agricultural_data[['Soil_Moisture', 'Nutrient_Level', 'Crop_Health', 'Temperature', 'Humidity', 'Precipitation']]
y = agricultural_data['Yield_Percentage']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a RandomForestRegressor model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
predictions = model.predict(X_test)
mse = mean_squared_error(y_test, predictions)
print(f'Mean Squared Error: {mse}')

# Implement variable rate application of inputs based on model predictions
# Example: Adjust application rates of water, fertilizers, and pesticides based on predicted crop yields and localized crop needs
def adjust_application_rates(predictions, crop_needs):
    # Example implementation of adjusting application rates based on predictions and crop needs
    adjusted_water_rate = predictions * crop_needs['Water']
    adjusted_fertilizer_rate = predictions * crop_needs['Fertilizer']
    adjusted_pesticide_rate = predictions * crop_needs['Pesticide']
    return adjusted_water_rate, adjusted_fertilizer_rate, adjusted_pesticide_rate

# Example crop needs (can be adjusted based on specific crop requirements)
crop_needs = {'Water': 0.7, 'Fertilizer': 0.5, 'Pesticide': 0.3}

# Example prediction for a specific area
area_prediction = model.predict([[0.6, 0.4, 0.8, 30, 60, 0]])
adjusted_water_rate, adjusted_fertilizer_rate, adjusted_pesticide_rate = adjust_application_rates(area_prediction, crop_needs)
print(f'Adjusted Water Rate: {adjusted_water_rate}')
print(f'Adjusted Fertilizer Rate: {adjusted_fertilizer_rate}')
print(f'Adjusted Pesticide Rate: {adjusted_pesticide_rate}')
