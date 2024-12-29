import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.impute import SimpleImputer

# Function to fetch weather data for each circuit
def get_weather_data_for_circuits(data_merged):
    weather_info = []
    
    # Replace with actual API call for weather data
    for race_id, circuit in data_merged.iterrows():
        city = circuit['city']
        weather_data = fetch_weather_data(city)
        if weather_data is not None:
            weather_info.append(weather_data)
        else:
            print(f"Error fetching weather data for {city}")
    
    weather_df = pd.DataFrame(weather_info)
    
    # Convert precipitation to mm
    weather_df['precipitation'] = weather_df['precipitation'].apply(convert_precipitation_to_mm)
    
    return pd.merge(data_merged, weather_df, on='race_id', how='left')

# Fetch weather data for a given city (replace this with an actual API)
def fetch_weather_data(city):
    # Example mockup weather data. Replace with actual API call in production.
    return {
        'race_id': 1,  # Example, link to actual race
        'temperature': np.random.rand() * 30 + 10,  # Random temp between 10-40C
        'humidity': np.random.rand() * 100,         # Random humidity between 0-100%
        'wind_speed': np.random.rand() * 15,        # Random wind speed between 0-15 m/s
        'precipitation': np.random.choice(['clear sky', 'overcast clouds', 'light rain', 'heavy rain'])  # Weather conditions
    }

# Convert precipitation to mm
def convert_precipitation_to_mm(precipitation):
    if precipitation in ['clear sky', 'few clouds', 'scattered clouds', 'broken clouds', 'overcast clouds']:
        return 0  # No rain
    elif precipitation == 'light rain':
        return np.random.uniform(1, 2)  # Light rain (1-2 mm/hour)
    elif precipitation == 'moderate rain':
        return np.random.uniform(3, 7)  # Moderate rain (3-7 mm/hour)
    elif precipitation == 'heavy rain':
        return np.random.uniform(8, 15)  # Heavy rain (8-15 mm/hour)
    elif precipitation == 'thunderstorm':
        return np.random.uniform(20, 40)  # Thunderstorm (20-40 mm/hour)
    else:
        return 0  # Default to no rain if unknown

# Fetch driver data using the Ergast API (returns driver info for the given driver_id)
def fetch_driver_data(driver_id):
    url = f"https://ergast.com/api/f1/drivers/{driver_id}.json"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        
        # Print the response for debugging
        print(f"Fetched data for driver_id: {driver_id}")
        print(data)

        # Check if the driver exists in the response
        try:
            driver_data = data['MRData']['DriverTable']['Drivers'][0]
            return {
                'driver_id': driver_data['driverId'],
                'given_name': driver_data['givenName'],
                'family_name': driver_data['familyName'],
                'nationality': driver_data['nationality'],
                'date_of_birth': driver_data['dateOfBirth'],
                'url': driver_data['url']
            }
        except (IndexError, KeyError) as e:
            print(f"No data found for driver_id: {driver_id}. Error: {e}")
            return None
    else:
        print(f"Error fetching data for driver_id: {driver_id}, Status code: {response.status_code}")
        return None

# Main function to orchestrate data loading, merging, and modeling
def main():
    # Load data from CSV files
    lap_times = pd.read_csv('lap_times.csv')
    drivers = pd.read_csv('drivers.csv')  # Contains the driver_id to be used in API call
    circuits = pd.read_csv('circuits.csv')  # Contains circuit_id and city info

    # Strip any leading/trailing spaces in the column names
    lap_times.columns = lap_times.columns.str.strip()
    drivers.columns = drivers.columns.str.strip()
    circuits.columns = circuits.columns.str.strip()

    print(f"Lap Times Columns: {lap_times.columns}")
    print(f"Driver Data Columns: {drivers.columns}")
    print(f"Circuit Data Columns: {circuits.columns}")

    # Merge lap times, driver, and circuit data
    if 'driver_id' in lap_times.columns:
        data_merged = pd.merge(lap_times, drivers, on='driver_id', how='left')
        data_merged = pd.merge(data_merged, circuits, on='circuit_id', how='left')
    else:
        print("driver_id column is missing in the lap_times data.")
        return
    
    # Fetch driver data using Ergast API for each driver_id
    driver_data_list = []
    for driver_id in data_merged['driver_id'].unique():
        driver_data = fetch_driver_data(driver_id)
        if driver_data:
            driver_data_list.append(driver_data)

    # Debug print to inspect the collected driver data
    print(f"Collected Driver Data: {driver_data_list}")

    # Convert the list to a DataFrame and merge it with existing data
    driver_df = pd.DataFrame(driver_data_list)
    print(f"Driver DataFrame: {driver_df.head()}")

    # Merge the driver data into the existing dataset
    if not driver_df.empty:
        data_merged = pd.merge(data_merged, driver_df, on='driver_id', how='left')
    else:
        print("No driver data available to merge.")

    # Fetch weather data for each circuit and merge it with the existing data
    data_with_weather = get_weather_data_for_circuits(data_merged)
    
    print(f"Data with Weather: {data_with_weather.head()}")

    # Drop rows with NaN values in any of the features (or impute them)
    data_with_weather = data_with_weather.dropna(subset=['temperature', 'humidity', 'wind_speed', 'precipitation'])
    
    # Alternatively, you can use an imputer to fill missing values:
    # imputer = SimpleImputer(strategy='mean')
    # data_with_weather[['temperature', 'humidity', 'wind_speed', 'precipitation']] = imputer.fit_transform(data_with_weather[['temperature', 'humidity', 'wind_speed', 'precipitation']])

    # Prepare features and target variable for modeling
    X = data_with_weather[['temperature', 'humidity', 'wind_speed', 'precipitation']]
    y = data_with_weather['lap_time_wet']  # Assuming you want to predict lap times in wet conditions

    # Train a regression model
    model = LinearRegression()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
    model.fit(X_train, y_train)

    # Predict and evaluate the model
    y_pred = model.predict(X_test)
    
    # Calculate performance metrics
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Mean Absolute Error (MAE): {mae}")
    print(f"R² Score: {r2}")
    
    # Visualization: Scatter plot of predictions vs actual values
    plt.scatter(y_test, y_pred)
    plt.xlabel("Actual Lap Time (Wet)")
    plt.ylabel("Predicted Lap Time (Wet)")
    plt.title("Actual vs Predicted Lap Times (Wet Conditions)")
    plt.show()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Exception occurred: {e}")
