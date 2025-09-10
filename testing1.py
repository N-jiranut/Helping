import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# The list of indices to increase
incress = [6,7,8,9,10,17,18,19,20,21,28,29,30,31,32,39,40,41,42,43,50,51,52,53,54,61,62,63,64,65,72,73,74,75,76,83,84,85,86,87,94,95,96,97,98,105,106,107,108,109]

# The raw data, still as strings
forpredict = ['0.35', '3.88', '8.54', '0.53', '-0.3', '-0.21', '202', '318', '539', '637', '46', '0.1', '-2.65', '9.97', '-0.07', '-0.06', '0.04', '569', '741', '661', '623', '493', '-2.89', '2.9', '8.69', '0.01', '0.02', '0.01', '201', '340', '600', '416', '45', '0.25', '-2.63', '9.76', '-0.08', '-0.08', '0.03', '565', '741', '664', '631', '493', '-3.04', '3.01', '8.75', '0', '-0.05', '0.01', '199', '341', '596', '423', '19', '0.21', '-2.7', '9.96', '-0.07', '-0.02', '0.03', '566', '734', '666', '611', '493', '-3.05', '2.64', '10.23', '0.22', '0.1', '-0.08', '199', '328', '601', '445', '37', '-0.05', '-2.84', '9.79', '-0.07', '0.01', '0.04', '1385', '2156', '667', '361', '491', '-7.76', '3.54', '3.65', '1.24', '0.85', '-1.57', '217', '255', '485', '655', '69', '-0.2', '-2.85', '9.74', '-0.06', '0.02', '0.04', '567', '740', '669', '613', '491', '-7.76', '3.54', '3.65', '1.24', '0.85', '-1.57', '217', '255', '485', '655', '69', '-0.27', '-2.95', '9.78', '-0.06', '0', '0.04', '565', '741', '676', '615', '491'] 

# Create a DataFrame from the list, with one row and multiple columns
# Note: The data is currently a list of strings, which is fine for the scaler.
df = pd.DataFrame([forpredict])

# Transpose the DataFrame to have one column and multiple rows, which is the
# format the scaler expects.
dfx = df.T

# Initialize the scaler and fit it to the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_datat = scaler.fit_transform(dfx)

# Transpose the scaled data back to a single row
# The shape is now (1, 100)
scaled_data = scaled_datat.T

# IMPORTANT: The IndexError occurs because scaled_data is a 2D array with a shape of (1, 100).
# You were trying to access a row (e.g., scaled_data[6]), but there is only one row at index 0.
# To access the values, you need to first select the row (index 0) and then the column (index n).
for n in incress:
    # We access the single row with [0] and then the specific column with [n].
    scaled_data[0][n] = scaled_data[0][n] * 2

# Convert the NumPy array back to a list
forpredict = scaled_data[0].tolist()

# Convert the list to a NumPy array for the final output
forpredict = np.array(forpredict)
print(forpredict)