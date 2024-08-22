import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Load the dataset
df = pd.read_csv("./models/water_requirement_model.csv")

# Separate numeric and categorical data
my_numeric_df = df.select_dtypes(exclude='object')
my_object_df = df.select_dtypes(include='object')

# Convert categorical data to dummy variables
df_object_dummies = pd.get_dummies(my_object_df, drop_first=True)
df_object_dummies = df_object_dummies.astype(int)
final_df = pd.concat([my_numeric_df, df_object_dummies], axis=1)

# Split data into features and target variable
X = final_df.drop(columns=['WATER REQUIREMENT'])
y = final_df['WATER REQUIREMENT']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Random Forest Regressor model
model = RandomForestRegressor(n_estimators=100, min_samples_leaf=4, min_samples_split=10)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate mean squared error
mse = mean_squared_error(y_test, y_pred)
print('Random Forest Regressor Mean Squared Error: {0:0.4f}'.format(mse))

# Hardcoded input values
input_data_dict = {
    'CROP TYPE_Wheat': 1,
    'SOIL TYPE_Loamy': 1,
    'REGION_North': 1,
    'TEMPERATURE_30-40': 1,
    'WEATHER CONDITION_Sunny': 1
}

# Ensure all columns from training data are present in input_data
input_data = pd.DataFrame(columns=X_train.columns)
for col in X_train.columns:
    input_data[col] = [input_data_dict.get(col, 0)]  # Fill with 0 if not in input_data_dict

# Predict water requirement
predicted_water_requirement = model.predict(input_data)[0]
print(f"Predicted Water Requirement: {predicted_water_requirement:.2f} units")
