from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from flask_cors import CORS
from sklearn.metrics import accuracy_score

app = Flask(__name__)
CORS(app)

# Load the datasets
cropdf = pd.read_csv("./models/Crop_recommendation.csv")
water_df = pd.read_csv("./models/water_requirement_model.csv")
data = pd.read_csv('./models/ICRISAT.csv')
price_data = pd.read_csv('./models/ICRISAT_PRIZE.csv')

# Prepare the data for crop recommendation
X = cropdf.drop('label', axis=1)
y = cropdf['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True, random_state=0)

# Train the LightGBM model for crop recommendation
model = lgb.LGBMClassifier()
model.fit(X_train, y_train)

# Prepare the data for water requirement prediction
my_numeric_df = water_df.select_dtypes(exclude='object')
my_object_df = water_df.select_dtypes(include='object')
df_object_dummies = pd.get_dummies(my_object_df, drop_first=True)
df_object_dummies = df_object_dummies.astype(int)
final_df = pd.concat([my_numeric_df, df_object_dummies], axis=1)
X = final_df.drop(columns=['WATER REQUIREMENT'])
y = final_df['WATER REQUIREMENT']
X_train_water, X_test_water, y_train_water, y_test_water = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Random Forest Regressor model for water requirement prediction
water_model = RandomForestRegressor(n_estimators=100, min_samples_leaf=4, min_samples_split=10)
water_model.fit(X_train_water, y_train_water)

def predict_top_crops(model, sample_input, top_n=5):
    probas = model.predict_proba(sample_input)
    top_crops_list = []
    for probas_row in probas:
        top_indices = np.argsort(probas_row)[-top_n:][::-1]
        top_crops = [model.classes_[i] for i in top_indices]
        top_probas = probas_row[top_indices]
        top_crops_list.append(list(zip(top_crops, top_probas)))
    return top_crops_list

@app.route('/crop_recommendation', methods=['GET'])
def crop_recommendation():
    n = float(request.args.get('n'))
    p = float(request.args.get('p'))
    k = float(request.args.get('k'))
    temperature = float(request.args.get('temperature'))
    humidity = float(request.args.get('humidity'))
    ph = float(request.args.get('ph'))
    rainfall = float(request.args.get('rainfall'))

    sample_input = np.array([[n, p, k, temperature, humidity, ph, rainfall]])
    top_crops_list = predict_top_crops(model, sample_input, top_n=5)

    response = []
    for top_crops in top_crops_list:
        for crop, probability in top_crops:
            response.append({"crop": crop, "probability": float(probability)})

    return jsonify(response)

@app.route('/water_recommendation', methods=['GET'])
def water_recommendation():
    crop_type = request.args.get('crop_type')
    soil_type = request.args.get('soil_type')
    region = request.args.get('region')
    temperature = request.args.get('temperature')
    weather_condition = request.args.get('weather_condition')

    input_data_dict = {
        'CROP TYPE_Wheat': 1 if crop_type == 'Wheat' else 0,
        'SOIL TYPE_Loamy': 1 if soil_type == 'Loamy' else 0,
        'REGION_North': 1 if region == 'North' else 0,
        'TEMPERATURE_30-40': 1 if temperature == '30-40' else 0,
        'WEATHER CONDITION_Sunny': 1 if weather_condition == 'Sunny' else 0
    }

    input_data = pd.DataFrame(columns=X_train_water.columns)
    for col in X_train_water.columns:
        input_data[col] = [input_data_dict.get(col, 0)]

    predicted_water_requirement = water_model.predict(input_data)[0]
    response = {"water_requirement": float(predicted_water_requirement)}

    return jsonify(response)



# Function to get unique states
def get_states():
    return data['State Name'].unique().tolist()

# Function to filter districts based on the selected state
def filter_districts(selected_state):
    districts = data[data['State Name'] == selected_state]['Dist Name'].unique().tolist()
    return districts

# Function to filter available crops dynamically based on the selected state and district
def filter_crops(selected_state, selected_district):
    # Filter the data for the selected state and district
    filtered_data = data[(data['State Name'] == selected_state) & (data['Dist Name'] == selected_district)]

    # Find crop columns that have non-null data in the filtered dataset
    crop_columns = [col for col in data.columns if 'AREA' in col or 'PRODUCTION' in col or 'YIELD' in col]
    available_crops = [col for col in crop_columns if not filtered_data[col].dropna().empty]
    return available_crops

# Function to retrieve and display data points of all available data
def get_data_points(selected_state, selected_district, selected_crop):
    # Filter data for the selected district within the selected state
    filtered_data = data[(data['State Name'] == selected_state) & (data['Dist Name'] == selected_district)]

    # Ensure 'Year' and the selected crop column exist in the filtered data
    if 'Year' in filtered_data.columns and selected_crop in filtered_data.columns:
        crop_data = filtered_data[['Year', selected_crop]].dropna().reset_index(drop=True)
        if not crop_data.empty:
            return crop_data.values.tolist()  # Convert DataFrame to list of lists
        else:
            return "No data available for the selected crop."
    else:
        return "Year column or selected crop not found in dataset."


@app.route('/market_insights', methods=['GET'])
def market_insights():
    state = request.args.get('state')
    city = request.args.get('city')
    crop_type = request.args.get('crop_type')

    if not state or not city or not crop_type:
        return jsonify({"error": "Missing required parameters"}), 400

    # Get available options
    states = get_states()
    if state not in states:
        return jsonify({"error": "State not found"}), 404

    districts = filter_districts(state)
    if city not in districts:
        return jsonify({"error": "District not found"}), 404

    crops = filter_crops(state, city)
    if crop_type not in crops:
        return jsonify({"error": "Crop type not available"}), 404

    # Retrieve data points
    data_points = get_data_points(state, city, crop_type)
    if isinstance(data_points, str):
        return jsonify({"error": data_points}), 404

    # Prepare the response
    response = [{"year": year, "value": value} for year, value in data_points]
    return jsonify(response)


@app.route('/market_prize', methods=['GET'])
def market_prize():
    state = request.args.get('state')
    city = request.args.get('city')
    crop_type = request.args.get('crop_type')

    if not state or not city or not crop_type:
        return jsonify({"error": "Missing required parameters"}), 400

    # Get crop data
    result = get_crop_data(state, city, crop_type)
    if isinstance(result, list):
        response = [{"year": year, "price": price} for year, price in result]
        return jsonify(response)
    else:
        return jsonify({"error": result}), 404

# Helper functions
def get_crop_data(selected_state, selected_district, selected_crop):
    # Filter the price data based on the selected state and district
    filtered_data = price_data[(price_data['State Name'] == selected_state) & (price_data['Dist Name'] == selected_district)]

    if 'Year' in filtered_data.columns:
        # Extract the crop data
        if selected_crop in filtered_data.columns:
            crop_data = filtered_data[['Year', selected_crop]].dropna().reset_index(drop=True)

            # Remove rows where the price is -1 (not produced)
            crop_data = crop_data[crop_data[selected_crop] != -1]

            if crop_data.empty:
                return "No data available for the selected crop in this district."

            # Convert to pairs
            data_points = crop_data[['Year', selected_crop]].values.tolist()
            return data_points
        else:
            return "Selected crop not found in dataset."
    else:
        return "Year column not found in dataset."

if __name__ == '__main__':
    app.run(debug=True)
