import pandas as pd
import numpy as np
import folium
from folium.plugins import MarkerCluster
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# ==========================
# 1. Load and Preprocess Data
# ==========================
def load_and_preprocess_data(file_path):
    """Load and preprocess NYPD crime data."""
    data = pd.read_csv(file_path, low_memory=False)
    
    # Drop irrelevant columns
    columns_to_drop = [
        'CMPLNT_NUM', 'ADDR_PCT_CD', 'HADEVELOPT', 'HOUSING_PSA', 'JURISDICTION_CODE', 'JURIS_DESC',
        'PARKS_NM', 'PATROL_BORO', 'PD_CD', 'PD_DESC', 'STATION_NAME',
        'TRANSIT_DISTRICT', 'Lat_Lon', 'New Georeferenced Column'
    ]
    data.drop(columns=columns_to_drop, inplace=True, errors='ignore')
    
    # Drop rows missing essential info
    data.dropna(subset=['Latitude', 'Longitude', 'OFNS_DESC', 'CMPLNT_FR_TM'], inplace=True)
    
    # Convert dates
    data['CMPLNT_FR_DT'] = pd.to_datetime(data['CMPLNT_FR_DT'], format='%m/%d/%Y', errors='coerce')

    # Convert times
    data['CMPLNT_FR_TM'] = pd.to_datetime(data['CMPLNT_FR_TM'], format='%H:%M:%S', errors='coerce')

    # Create time-based features
    data['Month'] = data['CMPLNT_FR_DT'].dt.month
    data['DayOfWeek'] = data['CMPLNT_FR_DT'].dt.weekday
    data['Hour'] = data['CMPLNT_FR_TM'].dt.hour  
    
    # Drop rows where Hour could not be extracted
    data.dropna(subset=['Hour'], inplace=True)
    data['Hour'] = data['Hour'].astype(int)

    return data

# ==========================
# 2. Prepare Data
# ==========================
def prepare_binary_data(data):
    """Prepare data for binary classification."""
    data['Crime_Occurred'] = 1  # Real crimes

    # Simulate 'no crime' points
    no_crime_points = data.sample(n=len(data)//2).copy()
    no_crime_points['Latitude'] += np.random.uniform(-0.01, 0.01, size=no_crime_points.shape[0])
    no_crime_points['Longitude'] += np.random.uniform(-0.01, 0.01, size=no_crime_points.shape[0])
    no_crime_points['Crime_Occurred'] = 0
    no_crime_points['OFNS_DESC'] = 'No Crime'

    combined_data = pd.concat([data, no_crime_points], ignore_index=True)

    features = ['Latitude', 'Longitude', 'Hour', 'DayOfWeek', 'Month']
    X = combined_data[features]
    y = combined_data['Crime_Occurred']

    return train_test_split(X, y, test_size=0.3, random_state=42), combined_data

def prepare_multiclass_data(data):
    """Prepare data for multiclass classification."""
    features = ['Latitude', 'Longitude', 'Hour', 'DayOfWeek', 'Month']
    X = data[features]
    y = data['OFNS_DESC']
    return train_test_split(X, y, test_size=0.3, random_state=42)

# ==========================
# 3. Train and Evaluate Models
# ==========================
def train_random_forest(X_train, y_train):
    """Train a Random Forest Classifier."""
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test, label_desc):
    """Evaluate the model."""
    y_pred = model.predict(X_test)
    print(f"\n====== {label_desc} ======")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

# ==========================
# 4. Map Functions
# ==========================
def map_crime_occurrence(data):
    """Map predicted crime occurrence."""
    crime_map = folium.Map(location=[40.7128, -74.0060], zoom_start=12)
    cluster = MarkerCluster().add_to(crime_map)

    for _, row in data.iterrows():
        lat, lon = row['Latitude'], row['Longitude']
        crime_pred = row['Predicted_Crime_Occurred']

        color = 'red' if crime_pred == 1 else 'green'
        popup_text = 'Crime Occurred' if crime_pred == 1 else 'No Crime'

        folium.Marker(
            location=[lat, lon],
            popup=popup_text,
            icon=folium.Icon(color=color, icon='info-sign')
        ).add_to(cluster)
    
    folium.LayerControl().add_to(crime_map)
    crime_map.save("crime_occurrence_map.html")
    print("Crime occurrence map saved as 'crime_occurrence_map.html'.")

def map_crime_types(data):
    """Map predicted crime types."""
    crime_type_map = folium.Map(location=[40.7128, -74.0060], zoom_start=12)
    cluster = MarkerCluster().add_to(crime_type_map)

    for _, row in data.iterrows():
        lat, lon = row['Latitude'], row['Longitude']
        crime_type = row['Predicted_OFNS_DESC']

        folium.Marker(
            location=[lat, lon],
            popup=f"Predicted Crime Type: {crime_type}",
            icon=folium.Icon(color='blue', icon='info-sign')
        ).add_to(cluster)

    folium.LayerControl().add_to(crime_type_map)
    crime_type_map.save("crime_type_map.html")
    print("Crime type map saved as 'crime_type_map.html'.")

# ==========================
# 5. Backtesting
# ==========================
def backtest_model(data, features, model, label):
    """Backtest model month-by-month."""
    months = sorted(data['Month'].unique())
    accuracies = []

    print("\n===== BACKTESTING BY MONTH =====")
    for month in months:
        month_data = data[data['Month'] == month]
        if len(month_data) < 10:
            continue  # Skip months with very few samples

        X_month = month_data[features]
        y_month = month_data[label]

        y_pred = model.predict(X_month)
        acc = accuracy_score(y_month, y_pred)
        accuracies.append((month, acc))
        print(f"Month {month}: Accuracy = {acc:.4f}")

    return accuracies

# ==========================
# 6. Bias Testing
# ==========================
def simple_income_bias_test(binary_data, census_data):
    """Test if model predicts more crime in lower-income areas vs higher-income areas."""
    print("\n====== Simple Bias Test: Income vs Crime Prediction ======")

    # Merge on Borough name
    # Normalize names before merging
    binary_data['BORO_NM'] = binary_data['BORO_NM'].str.strip().str.upper()
    census_data['Borough'] = census_data['Borough'].str.strip().str.upper()

    merged = binary_data.merge(
        census_data[['Borough', 'Income']],
        left_on='BORO_NM',
        right_on='Borough',
        how='left'
)

    # Drop rows without income info
    merged = merged.dropna(subset=['Income'])

    # Split into Low vs High income areas
    income_median = merged['Income'].median()
    merged['Income_Group'] = np.where(merged['Income'] < income_median, 'Low Income', 'High Income')

    # Calculate crime rates
    group_stats = merged.groupby('Income_Group').agg({
        'Crime_Occurred': 'mean',
        'Predicted_Crime_Occurred': 'mean',
        'Latitude': 'count'
    }).rename(columns={'Latitude': 'Num_Samples'})

    print(group_stats)

    # Calculate bias
    group_stats['Prediction_Bias'] = group_stats['Predicted_Crime_Occurred'] - group_stats['Crime_Occurred']
    print("\nPrediction bias by income group:\n", group_stats[['Prediction_Bias']])

# ==========================
# 7. Main Pipeline
# ==========================
def main(file_path):
    """Execute the crime prediction pipeline."""
    data = load_and_preprocess_data(file_path)
    
    # Binary classification (crime occurrence)
    (Xb_train, Xb_test, yb_train, yb_test), binary_data = prepare_binary_data(data)
    binary_model = train_random_forest(Xb_train, yb_train)
    evaluate_model(binary_model, Xb_test, yb_test, "Binary Classification (Crime Occurrence)")

    binary_features = ['Latitude', 'Longitude', 'Hour', 'DayOfWeek', 'Month']
    binary_data['Predicted_Crime_Occurred'] = binary_model.predict(binary_data[binary_features])
    map_crime_occurrence(binary_data)

    # Multiclass classification (crime type)
    Xm_train, Xm_test, ym_train, ym_test = prepare_multiclass_data(data)
    multiclass_model = train_random_forest(Xm_train, ym_train)
    evaluate_model(multiclass_model, Xm_test, ym_test, "Multiclass Classification (Crime Type)")

    data['Predicted_OFNS_DESC'] = multiclass_model.predict(data[binary_features])
    map_crime_types(data)

    # Run backtesting
    backtest_model(binary_data, binary_features, binary_model, 'Crime_Occurred')
    backtest_model(data, binary_features, multiclass_model, 'OFNS_DESC')

    # Run simple income bias test
    census_data = pd.read_csv('./NYC_Census_Data.csv')
    simple_income_bias_test(binary_data, census_data)

    return data, binary_model, multiclass_model

# ==========================
# 8. Run the Script
# ==========================
if __name__ == '__main__':
    file_path = './NYPD_Complaint_Data_Current__Year_To_Date__20250226.csv'
    
    data, binary_model, multiclass_model = main(file_path)
