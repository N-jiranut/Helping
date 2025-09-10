import pandas as pd
from sklearn.preprocessing import MinMaxScaler

incress = ['l_accel_x_0', 'l_accel_y_0', 'l_accel_z_0', 'l_gyro_x_0', 'l_gyro_y_0', 'l_gyro_z_0', 'r_accel_x_0', 'r_accel_y_0', 'r_accel_z_0', 'r_gyro_x_0', 'r_gyro_y_0', 'r_gyro_z_0', 'l_accel_x_1', 'l_accel_y_1', 'l_accel_z_1', 'l_gyro_x_1', 'l_gyro_y_1', 'l_gyro_z_1', 'r_accel_x_1', 'r_accel_y_1', 'r_accel_z_1', 'r_gyro_x_1', 'r_gyro_y_1', 'r_gyro_z_1', 'l_accel_x_2', 'l_accel_y_2', 'l_accel_z_2', 'l_gyro_x_2', 'l_gyro_y_2', 'l_gyro_z_2', 'r_accel_x_2', 'r_accel_y_2', 'r_accel_z_2', 'r_gyro_x_2', 'r_gyro_y_2', 'r_gyro_z_2', 'l_accel_x_3', 'l_accel_y_3', 'l_accel_z_3', 'l_gyro_x_3', 'l_gyro_y_3', 'l_gyro_z_3', 'r_accel_x_3', 'r_accel_y_3', 'r_accel_z_3', 'r_gyro_x_3', 'r_gyro_y_3', 'r_gyro_z_3', 'l_accel_x_4', 'l_accel_y_4', 'l_accel_z_4', 'l_gyro_x_4', 'l_gyro_y_4', 'l_gyro_z_4', 'r_accel_x_4', 'r_accel_y_4', 'r_accel_z_4', 'r_gyro_x_4', 'r_gyro_y_4', 'r_gyro_z_4', 'l_accel_x_5', 'l_accel_y_5', 'l_accel_z_5', 'l_gyro_x_5', 'l_gyro_y_5', 'l_gyro_z_5', 'r_accel_x_5', 'r_accel_y_5', 'r_accel_z_5', 'r_gyro_x_5', 'r_gyro_y_5', 'r_gyro_z_5']

# Load CSV
df = pd.read_csv('data/main.csv', header=None)  # read without headers

# Use the first row as header
new_header = df.iloc[0]      # first row
df = df[1:]                  # data without first row
df.columns = new_header      # set first row as header

# Optionally, convert all data to numeric (in case CSV is strings)
df = df.apply(pd.to_numeric)

# Transpose, scale, then transpose back
dfn = df.T
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dfn)
scaled_df = pd.DataFrame(scaled_data.T, columns=df.columns, index=df.index)
# Increase specified columns by a factor of 2

# for col in incress:
#     scaled_df[col] = scaled_df[col] * 10

# Save CSV, keep the scaled first row as header
scaled_df.to_csv('data/main_scaled.csv', index=False, header=True)