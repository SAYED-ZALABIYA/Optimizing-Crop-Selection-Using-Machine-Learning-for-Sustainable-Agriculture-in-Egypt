import pandas as pd
import numpy as np

# Data Preprocessing
df2 = df.drop(columns=['Crop'])
numeric_columns = ['Nitrogen (kg/ha )', 'Phosphorus (kg/ha )', 'Potassium (kg/ha )',  'Temperature', 'Humidity', 'pH_Value', 'Rainfall']

# Scaling numeric features
scaler = MinMaxScaler()
df2[numeric_columns] = scaler.fit_transform(df2[numeric_columns].astype(np.float32))

# Features and target
x = df2
y = df['Crop']