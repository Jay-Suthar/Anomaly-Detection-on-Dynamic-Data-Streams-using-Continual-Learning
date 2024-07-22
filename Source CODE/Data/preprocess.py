import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

# Step 1: Load the Dataset
df = pd.read_csv('chicago.csv')

# Step 2: Determine Top Departments
tau = 15
top_departments = df['DEPARTMENT NAME'].value_counts().nlargest(tau).index.tolist()

# Step 3: Random Sampling
eta = 10000
sampled_payments = pd.concat([df[df['DEPARTMENT NAME'] == dept].sample(eta, replace=True).sort_values(by='CHECK DATE') for dept in top_departments])

# Step 4: Transform Attribute Values
categorical_columns = ['VOUCHER NUMBER', 'CHECK DATE', 'DEPARTMENT NAME', 'CONTRACT NUMBER', 'VENDOR NAME', 'CASHED']
numerical_columns = ['AMOUNT']

# One-hot encode categorical columns
encoder = OneHotEncoder()
encoded_categorical = encoder.fit_transform(sampled_payments[categorical_columns])

# Scale numerical columns
scaler = MinMaxScaler()
scaled_numerical = scaler.fit_transform(sampled_payments[numerical_columns])

# Concatenate encoded categorical columns and scaled numerical columns
transformed_data = np.concatenate([encoded_categorical.toarray(), scaled_numerical], axis=1)

# Convert the transformed data into a DataFrame
transformed_df = pd.DataFrame(transformed_data, columns=encoder.get_feature_names_out().tolist() + numerical_columns)

transformed_df.to_csv('chicago.csv', index=False)
