import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import folium
from folium.plugins import MarkerCluster
import cProfile
import pstats

# ==========================
# 1. Load and Preprocess Data
# ==========================
def load_and_preprocess_data(file_path):
    """Load and preprocess NYPD crime data."""
    
    # Load data, specify low_memory=False to avoid dtype issues
    data = pd.read_csv(file_path, low_memory=False)
    
    # Check for column names to ensure they are correctly loaded
    print(f"Columns in the dataset: {data.columns}")
    
    # Drop irrelevant columns that won't be used for modeling
    columns_to_drop = [
        'HADEVELOPT', 'HOUSING_PSA', 'JURISDICTION_CODE', 'JURIS_DESC',
        'PARKS_NM', 'PATROL_BORO', 'PD_CD', 'PD_DESC', 'STATION_NAME',
        'TRANSIT_DISTRICT', 'Lat_Lon', 'New Georeferenced Column'
    ]
    data.drop(columns=columns_to_drop, inplace=True, errors='ignore')
    
    # Handle missing data (you can either drop rows or fill in missing values)
    # Dropping rows where essential data (e.g., 'Latitude', 'Longitude', 'OFNS_DESC') is missing
    data.dropna(subset=['Latitude', 'Longitude', 'OFNS_DESC'], inplace=True)
    
    # Convert 'CMPLNT_FR_DT' to datetime format
    data['CMPLNT_FR_DT'] = pd.to_datetime(data['CMPLNT_FR_DT'], format='%m/%d/%Y', errors='coerce')
    
    # Check for invalid date entries after conversion
    print(f"Rows with invalid 'CMPLNT_FR_DT' after conversion: {data['CMPLNT_FR_DT'].isna().sum()}")

    # Create additional features from the 'CMPLNT_FR_DT' datetime column
    data['Month'] = data['CMPLNT_FR_DT'].dt.month
    data['DayOfWeek'] = data['CMPLNT_FR_DT'].dt.weekday
    data['Hour'] = data['CMPLNT_FR_DT'].dt.hour

    # You can also extract more features if needed, such as year or day of year
    data['Year'] = data['CMPLNT_FR_DT'].dt.year
    data['DayOfYear'] = data['CMPLNT_FR_DT'].dt.dayofyear
    
    # You might want to encode categorical columns here if necessary (like 'OFNS_DESC', 'LAW_CAT_CD', etc.)
    # Example: Encoding 'OFNS_DESC' (if it's categorical)
    data['OFNS_DESC'] = data['OFNS_DESC'].astype('category')
    data['OFNS_DESC_encoded'] = data['OFNS_DESC'].cat.codes
    
    return data.sample(n=10)


def apply_kmeans(data, k):
    """Apply KMeans clustering to the data."""
    # Ensure there are no missing values in the columns to be used for KMeans
    data_clean = data.dropna(subset=['Latitude', 'Longitude', 'Month', 'DayOfWeek', 'Hour'])  # Adjust as necessary

    # Define the features for clustering (ensure all are numeric)
    numerical_features = ['Latitude', 'Longitude', 'Month', 'DayOfWeek', 'Hour']
    X = data_clean[numerical_features]

    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Apply KMeans
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10, init='k-means++')
    data_clean['Cluster'] = kmeans.fit_predict(X_scaled)

    return data_clean, X_scaled


# ==========================
# 3. Visualize Clusters
# ==========================
def visualize_clusters(data):
    """Visualize crime clusters using longitude and latitude."""
    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        x='Longitude', y='Latitude', hue='Cluster', data=data, palette='viridis', s=50
    )
    plt.title('Crime Clusters by Location')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.legend(title='Cluster')
    plt.show()


# ==========================
# 4. Prepare Data for Random Forest
# ==========================
def prepare_rf_data(data):
    """Prepare feature matrix and target variable for Random Forest."""
    # Define features (including cluster labels) and target
    X = data[['Latitude', 'Longitude', 'Hour', 'DayOfWeek', 'Month', 'Cluster']]
    y = data['OFNS_DESC']  # Target: Crime type

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    return X_train, X_test, y_train, y_test


# ==========================
# 5. Train and Evaluate Random Forest
# ==========================
def train_and_evaluate_rf(X_train, X_test, y_train, y_test):
    """Train Random Forest and evaluate model performance."""
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)

    # Make predictions
    y_pred = rf_model.predict(X_test)

    # Evaluate performance
    print("Model Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    return rf_model


# ==========================
# 6. Map Crime Data on a New York Map
# ==========================
def map_crime_data(data):
    """Map crime data over New York using Folium."""
    # Create a base map centered around New York City
    nyc_map = folium.Map(location=[40.7128, -74.0060], zoom_start=12)
    
    # Create a marker cluster
    marker_cluster = MarkerCluster().add_to(nyc_map)
    
    # Add markers to the cluster
    for idx, row in data.iterrows():
        lat = row['Latitude']
        lon = row['Longitude']
        crime_desc = row['OFNS_DESC']
        
        # Add a marker with popup showing crime description
        folium.Marker(
            location=[lat, lon],
            popup=f"Crime: {crime_desc}",
            icon=folium.Icon(color='red', icon='info-sign')
        ).add_to(marker_cluster)
    
    # Save the map to an HTML file
    nyc_map.save("nyc_crime_map.html")
    print("Crime map saved as 'nyc_crime_map.html'.")


# ==========================
# 7. Main Execution
# ==========================
def main(file_path, k=5):
    """Main function to execute the crime prediction pipeline."""
    # Load and preprocess data
    data = load_and_preprocess_data(file_path)

    # Apply K-Means clustering
    data, X_scaled = apply_kmeans(data, k)

    # Visualize Clusters
    visualize_clusters(data)

    # Prepare data for Random Forest
    X_train, X_test, y_train, y_test = prepare_rf_data(data)

    # Train and evaluate Random Forest
    rf_model = train_and_evaluate_rf(X_train, X_test, y_train, y_test)

    # Map crime data on New York map
    map_crime_data(data)

    return data, rf_model


# ==========================
# 8. Run the Pipeline with Profiling
# ==========================
if __name__ == '__main__':
    # Specify file path and number of clusters
    file_path = './NYPD_Complaint_Data_Current__Year_To_Date__20250226.csv'
    optimal_k = 5  # Adjust after reviewing the elbow method if necessary

    # Run the complete pipeline with profiling
    profiler = cProfile.Profile()
    profiler.enable()

    data, rf_model = main(file_path, k=optimal_k)

    profiler.disable()

    # Print profiling results
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumtime')  # Sort by cumulative time
    stats.print_stats(10)  # Print the top 10 time-consuming calls
