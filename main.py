import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load the data
data = pd.read_csv('./NYPD_Complaint_Data_Current__Year_To_Date__20250226.csv')

# Remove missing values
data.dropna(inplace=True)
data.pop('CMPLNT_FR_DT')
data.pop('CMPLNT_TO_DT')
data.pop('CRM_ATPT_CPTD_CD')
data.pop('HADEVELOPT')
data.pop('HOUSING_PSA')
data.pop('JURISDICTION_CODE')
data.pop('JURIS_DESC')
data.pop('PARKS_NM')
data.pop('PATROL_BORO')
data.pop('PD_CD')
data.pop('PD_DESC')
data.pop('RPT_DT')
data.pop('STATION_NAME')
data.pop('TRANSIT_DISTRICT')

# Encode categorical variables
le = LabelEncoder()
print(data.keys())
# data['Gender'] = le.fit_transform(data['Gender'])
# data['Family_History'] = le.fit_transform(data['Family_History'])

# Scale numerical variables
# scaler = StandardScaler()
# data[['Age', 'BMI', 'Glucose', 'Insulin']] = scaler.fit_transform(data[['Age', 'BMI', 'Glucose', 'Insulin']])