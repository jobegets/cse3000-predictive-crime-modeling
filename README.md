# Crime Prediction and Analysis

This project predicts crime occurrences and crime types based on NYPD Complaint Data using machine learning models. It also includes bias testing to check the fairness of the predictions based on socio-economic factors (like income) and geographical features (like time and location).

## Requirements

pandas for data manipulation

numpy for numerical operations

folium for interactive maps

scikit-learn for model building and evaluation

To install the required packages:

```bash
pip install pandas numpy folium scikit-learn
```

## How to Run

1. Ensure you have the NYPD_Complaint_Data and NYC_Census_Data CSV files.

The complaint data can be found here: https://data.cityofnewyork.us/Public-Safety/NYPD-Complaint-Data-Current-Year-To-Date-/5uac-w243/about_data

The census data can be found here (use nyc_census_tracts.csv): https://www.kaggle.com/datasets/muonneutrino/new-york-city-census-data/data

2. Run the Script:

```bash
python3 crime_prediction.py
```

### Notes

- The model's predictions are based on neutral features like location and time.
- Our report was concluded with the first 10000 rows of the NYC Complaint Data
