import pandas as pd
import folium
from folium.plugins import MarkerCluster
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import cProfile

# ==========================
# 1. Load and Preprocess Data
# ==========================
def load_and_preprocess_data(file_path):
    """Load and preprocess NYPD crime data."""
    data = pd.read_csv(file_path, low_memory=False)
    
    # Drop irrelevant columns
    columns_to_drop = [
        'HADEVELOPT', 'HOUSING_PSA', 'JURISDICTION_CODE', 'JURIS_DESC',
        'PARKS_NM', 'PATROL_BORO', 'PD_CD', 'PD_DESC', 'STATION_NAME',
        'TRANSIT_DISTRICT', 'Lat_Lon', 'New Georeferenced Column'
    ]
    data.drop(columns=columns_to_drop, inplace=True, errors='ignore')
    
    # Drop rows missing essential info
    data.dropna(subset=['Latitude', 'Longitude', 'OFNS_DESC'], inplace=True)
    
    # Convert dates
    data['CMPLNT_FR_DT'] = pd.to_datetime(data['CMPLNT_FR_DT'], format='%m/%d/%Y', errors='coerce')

    # Create time-based features
    data['Month'] = data['CMPLNT_FR_DT'].dt.month
    data['DayOfWeek'] = data['CMPLNT_FR_DT'].dt.weekday
    data['Hour'] = data['CMPLNT_FR_DT'].dt.hour
    
    return data

# ==========================
# 2. Prepare Data for Random Forest
# ==========================
def prepare_rf_data(data):
    """Prepare features and target for Random Forest."""
    features = ['Latitude', 'Longitude', 'Hour', 'DayOfWeek', 'Month']
    X = data[features]
    y = data['OFNS_DESC']
    
    return train_test_split(X, y, test_size=0.3, random_state=42)

# ==========================
# 3. Train and Evaluate Random Forest
# ==========================
def train_and_evaluate_rf(X_train, X_test, y_train, y_test):
    """Train Random Forest and evaluate model."""
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    
    y_pred = rf_model.predict(X_test)
    
    print("Model Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    
    return rf_model

# ==========================
# 4. Map Crime Data
# ==========================
def map_crime_data(data):
    """Map actual and predicted crime types."""
    nyc_map = folium.Map(location=[40.7128, -74.0060], zoom_start=12)
    actual_cluster = MarkerCluster(name='Actual Crimes').add_to(nyc_map)
    predicted_cluster = MarkerCluster(name='Predicted Crimes').add_to(nyc_map)
    
    for _, row in data.iterrows():
        lat, lon = row['Latitude'], row['Longitude']
        
        # Actual
        folium.Marker(
            location=[lat, lon],
            popup=f"Actual Crime: {row['OFNS_DESC']}",
            icon=folium.Icon(color='red', icon='info-sign')
        ).add_to(actual_cluster)
        
        # Predicted (slight offset)
        folium.Marker(
            location=[lat + 0.0005, lon + 0.0005],
            popup=f"Predicted Crime: {row['Predicted_OFNS_DESC']}",
            icon=folium.Icon(color='blue', icon='info-sign')
        ).add_to(predicted_cluster)
    
    folium.LayerControl().add_to(nyc_map)
    nyc_map.save("nyc_crime_map.html")
    print("Crime map saved as 'nyc_crime_map.html'.")

# ==========================
# 5. Main Function
# ==========================
def main(file_path):
    """Execute the crime prediction pipeline."""
    data = load_and_preprocess_data(file_path)
    
    X_train, X_test, y_train, y_test = prepare_rf_data(data)
    rf_model = train_and_evaluate_rf(X_train, X_test, y_train, y_test)
    
    # Predict on full dataset
    features = ['Latitude', 'Longitude', 'Hour', 'DayOfWeek', 'Month']
    data['Predicted_OFNS_DESC'] = rf_model.predict(data[features])
    
    map_crime_data(data)
    return data, rf_model

# ==========================
# 6. Run the Script
# ==========================
if __name__ == '__main__':
    file_path = './NYPD_Complaint_Data_Current__Year_To_Date__20250226.csv'
    
    profiler = cProfile.Profile()
    profiler.enable()
    
    data, rf_model = main(file_path)
