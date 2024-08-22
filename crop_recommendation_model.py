import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the dataset
cropdf = pd.read_csv("./models/Crop_recommendation.csv")
print(cropdf.head())
print("List of crops: ", cropdf['label'].unique())

# Prepare the data
X = cropdf.drop('label', axis=1)
y = cropdf['label']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True, random_state=0)

# Train the LightGBM model
model = lgb.LGBMClassifier()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_pred, y_test)
print('LightGBM Model accuracy score: {0:0.4f}'.format(accuracy))

# Hardcoded input values (N, P, K, temperature, humidity, pH, rainfall)
sample_input = np.array([[20, 30, 40, 25.5, 80, 6.5, 100]])

print("Input data:")
print(sample_input)

def predict_top_crops(model, sample_input, top_n=5):
    probas = model.predict_proba(sample_input)
    top_crops_list = []
    for probas_row in probas:
        top_indices = np.argsort(probas_row)[-top_n:][::-1]
        top_crops = [model.classes_[i] for i in top_indices]
        top_probas = probas_row[top_indices]
        top_crops_list.append(list(zip(top_crops, top_probas)))
    return top_crops_list

# Predict the top 5 crops for the sample input
top_crops_list = predict_top_crops(model, sample_input, top_n=5)

print("\nTop 5 Preferred Crops as Per Input Data:")
for i, top_crops in enumerate(top_crops_list):
    for crop in top_crops:
        print(f"Crop: {crop[0]}")
    print()
