import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder

# Load the dataset
data_f = pd.read_csv('your_dataset.csv')

# Select relevant columns (skipping 'GDP per capita (current US$)' and 'Mortality rate, infant (per 1,000 live births)')
features = ['Year', 'Population, total', 'Unemployment, total (% of total labor force) (modeled ILO estimate)', 
            'Life expectancy at birth, total (years)', 'Country']
target = 'GDP (current US$)'

# Drop rows with missing target values
data_f = data_f.dropna(subset=[target])

# Handle missing values in features (e.g., fill with median)
data_f.fillna(data_f.median(), inplace=True)

# One-hot encode the 'Country' column
encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
encoded_countries = encoder.fit_transform(data_f[['Country']])
country_encoded_df = pd.DataFrame(encoded_countries, columns=encoder.get_feature_names_out(['Country']))

# Combine encoded features with the dataset
data_f = pd.concat([data_f, country_encoded_df], axis=1)
data_f.drop(['Country'], axis=1, inplace=True)

# Separate features (X) and target (y)
X = data_f[features]  # Only the selected features
y = data_f[target]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the model
model = RandomForestRegressor(random_state=42)

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error (MAE): {mae}")
print(f"Mean Squared Error (MSE): {mse}")
print(f"R² Score: {r2}")

# Example prediction for new data
new_data = pd.DataFrame({
    'Year': [2025],
    'Population, total': [200000000],
    'Unemployment, total (% of total labor force) (modeled ILO estimate)': [5.0],
    'Life expectancy at birth, total (years)': [70],
    **{col: 0 for col in encoder.get_feature_names_out(['Country'])}
})
new_data['Country_Pakistan'] = 1  # Set the country for the prediction
gdp_prediction = model.predict(new_data)
print(f"Predicted GDP: {gdp_prediction[0]}")
