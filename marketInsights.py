import pandas as pd

# Load the data
data = pd.read_csv('ICRISAT-District Level Data.csv')

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

# User input
state = input("Enter State: ")
district = input("Enter District: ")
crop = input("Enter Crop Type: ")

# Validate and get available options
states = get_states()
if state not in states:
    print("State not found.")
else:
    districts = filter_districts(state)
    if district not in districts:
        print("District not found.")
    else:
        crops = filter_crops(state, district)
        if crop not in crops:
            print("Crop type not available.")
        else:
            # Retrieve data points
            data_points = get_data_points(state, district, crop)
            print("Data Points:", data_points)

