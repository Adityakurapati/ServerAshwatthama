import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import imblearn
from imblearn.over_sampling import SMOTE
from collections import Counter

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import xgboost
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

# Load the dataset
data = pd.read_csv("Fertilizer Prediction Dataset.csv")

# Encode categorical data
soil_type_label_encoder = LabelEncoder()
data["Soil Type"] = soil_type_label_encoder.fit_transform(data["Soil Type"])

crop_type_label_encoder = LabelEncoder()
data["Crop Type"] = crop_type_label_encoder.fit_transform(data["Crop Type"])

fertname_label_encoder = LabelEncoder()
data["Fertilizer Name"] = fertname_label_encoder.fit_transform(data["Fertilizer Name"])

# Create mapping dictionaries for inverse transformation
croptype_dict = {i: crop_type_label_encoder.inverse_transform([i])[0] for i in range(len(data["Crop Type"].unique()))}
soiltype_dict = {i: soil_type_label_encoder.inverse_transform([i])[0] for i in range(len(data["Soil Type"].unique()))}
fertname_dict = {i: fertname_label_encoder.inverse_transform([i])[0] for i in range(len(data["Fertilizer Name"].unique()))}

# Define inverse mappings
soiltype_dict_inv = {v: k for k, v in soiltype_dict.items()}
croptype_dict_inv = {v: k for k, v in croptype_dict.items()}

# Split data into features and target variable
X = data.drop(columns=['Fertilizer Name'])
y = data['Fertilizer Name']

# Handle imbalanced data
counter = Counter(y)
upsample = SMOTE()
X, y = upsample.fit_resample(X, y)
counter = Counter(y)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X.values, y, test_size=0.2, random_state=0)

# Train different models and evaluate their performance
def train_and_evaluate_model(pipeline, model_name):
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"Accuracy on Test Data ({model_name}): {accuracy*100:.2f}%")
    return pipeline

# Training different models
knn_pipeline = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5))
knn_pipeline = train_and_evaluate_model(knn_pipeline, "KNN")

svm_pipeline = make_pipeline(StandardScaler(), SVC(probability=True))
svm_pipeline = train_and_evaluate_model(svm_pipeline, "SVM")

rf_pipeline = make_pipeline(StandardScaler(), RandomForestClassifier(random_state=18))
rf_pipeline = train_and_evaluate_model(rf_pipeline, "Random Forest")

xgb_pipeline = make_pipeline(StandardScaler(), XGBClassifier(random_state=18))
xgb_pipeline = train_and_evaluate_model(xgb_pipeline, "XGBoost")

# Use XGBoost for predictions (assuming highest accuracy)
final_model = knn_pipeline 

# Function to recommend fertilizer based on user inputs
def recommend_fertilizer(model, croptype_dict_inv, soiltype_dict_inv, fertname_dict):
    # Mock user inputs
    temperature = 25.0
    humidity = 50.0
    moisture = 20.0
    soil_type = soiltype_dict_inv["Loamy"]  # Example input
    crop_type = croptype_dict_inv["Wheat"]  # Example input
    nitrogen = 5.0
    potassium = 5.0
    phosphorous = 5.0

    # Prepare the input for the model
    user_input = np.array([[temperature, humidity, moisture, soil_type, crop_type, nitrogen, potassium, phosphorous]])

    input_features = ['Temperature', 'Humidity', 'Moisture', 'Soil Type', 'Crop Type',
                      'Nitrogen', 'Potassium', 'Phosphorous']
    print("\n\nUser Input:")
    for feature, value in zip(input_features, user_input[0]):
        print(f"{feature}: {value}")

    # Predict the fertilizer
    predicted_fertilizer = model.predict(user_input)[0]

    # Convert label to actual fertilizer name
    recommended_fertilizer = fertname_dict[predicted_fertilizer]
    print(f"Recommended Fertilizer: {recommended_fertilizer}")
    return recommended_fertilizer

# Predict and print the recommended fertilizer
recommended_fertilizer = recommend_fertilizer(final_model, croptype_dict_inv, soiltype_dict_inv, fertname_dict)