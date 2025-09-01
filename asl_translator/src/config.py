import os

# Project paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
MODEL_DIR = os.path.join(DATA_DIR, 'models')
LOG_DIR = os.path.join(PROJECT_ROOT, 'logs')

# Model settings
MODEL_CONFIG = {
    'n_estimators': 100,
    'max_depth': 20,
    'min_samples_split': 5,
    'min_samples_leaf': 2,
    'random_state': 42
}

# Real-time prediction settings
REALTIME_CONFIG = {
    'buffer_size': 50,
    'window_size': 30,
    'step_size': 15,
    'prediction_threshold': 0.7,
    'prediction_rate': 20  # Hz
}

# Communication settings
SERIAL_CONFIG = {
    'port': '/dev/ttyUSB0',  # Linux/Pi
    # 'port': 'COM3',        # Windows
    'baudrate': 115200,
    'timeout': 1
}

UDP_CONFIG = {
    'host': '0.0.0.0',
    'port': 8888
}

# Sensor configuration
SENSOR_COLUMNS = [
    'R_accel_x', 'R_accel_y', 'R_accel_z',
    'R_gyro_x', 'R_gyro_y', 'R_gyro_z', 'R_temp',
    'R_flex_1', 'R_flex_2', 'R_flex_3', 'R_flex_4', 'R_flex_5',
    'L_accel_x', 'L_accel_y', 'L_accel_z',
    'L_gyro_x', 'L_gyro_y', 'L_gyro_z', 'L_temp',
    'L_flex_1', 'L_flex_2', 'L_flex_3', 'L_flex_4', 'L_flex_5'
]

# Create directories if they don't exist
for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODEL_DIR, LOG_DIR]:
    os.makedirs(directory, exist_ok=True)