import numpy as np
import pandas as pd
import joblib
import threading
import queue
import time
from collections import deque
import json
import socket
import serial
from scipy.signal import butter, filtfilt
from scipy import stats
import logging

class RealTimeASLPredictor:
    def __init__(self, model_path, buffer_size=50, prediction_threshold=0.7):
        """
        Real-time ASL gesture predictor for Raspberry Pi
        
        Args:
            model_path: Path to trained model file
            buffer_size: Number of recent samples to keep in buffer
            prediction_threshold: Minimum confidence for gesture prediction
        """
        self.buffer_size = buffer_size
        self.prediction_threshold = prediction_threshold
        
        # Data buffers for each hand
        self.sensor_buffer = deque(maxlen=buffer_size)
        self.prediction_queue = queue.Queue()
        
        # Load trained model
        self.load_model(model_path)
        
        # Threading control
        self.running = False
        self.prediction_thread = None
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Sensor configuration
        self.sensor_columns = [
            'R_accel_x', 'R_accel_y', 'R_accel_z',
            'R_gyro_x', 'R_gyro_y', 'R_gyro_z', 'R_temp',
            'R_flex_1', 'R_flex_2', 'R_flex_3', 'R_flex_4', 'R_flex_5',
            'L_accel_x', 'L_accel_y', 'L_accel_z',
            'L_gyro_x', 'L_gyro_y', 'L_gyro_z', 'L_temp',
            'L_flex_1', 'L_flex_2', 'L_flex_3', 'L_flex_4', 'L_flex_5'
        ]
        
        self.last_prediction = {"gesture": "rest posture", "confidence": 1.0, "timestamp": time.time()}
    
    def load_model(self, model_path):
        """Load trained model and preprocessors"""
        try:
            model_data = joblib.load(model_path)
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.label_encoder = model_data['label_encoder']
            self.feature_columns = model_data['feature_columns']
            self.logger.info(f"Model loaded successfully from {model_path}")
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            raise
    
    def preprocess_sensor_data(self, raw_data):
        """
        Preprocess raw sensor data similar to training pipeline
        """
        df = pd.DataFrame([raw_data], columns=self.sensor_columns)
        
        # Create engineered features
        df = self._create_features(df)
        
        return df
    
    def _create_features(self, df):
        """Create the same engineered features as in training"""
        # Calculate magnitude for accelerometer and gyroscope
        df['R_accel_magnitude'] = np.sqrt(df['R_accel_x']**2 + df['R_accel_y']**2 + df['R_accel_z']**2)
        df['L_accel_magnitude'] = np.sqrt(df['L_accel_x']**2 + df['L_accel_y']**2 + df['L_accel_z']**2)
        df['R_gyro_magnitude'] = np.sqrt(df['R_gyro_x']**2 + df['R_gyro_y']**2 + df['R_gyro_z']**2)
        df['L_gyro_magnitude'] = np.sqrt(df['L_gyro_x']**2 + df['L_gyro_y']**2 + df['L_gyro_z']**2)
        
        # Calculate total flex sensor values for each hand
        df['R_flex_total'] = df[['R_flex_1', 'R_flex_2', 'R_flex_3', 'R_flex_4', 'R_flex_5']].sum(axis=1)
        df['L_flex_total'] = df[['L_flex_1', 'L_flex_2', 'L_flex_3', 'L_flex_4', 'L_flex_5']].sum(axis=1)
        
        # Calculate flex sensor variance
        df['R_flex_variance'] = df[['R_flex_1', 'R_flex_2', 'R_flex_3', 'R_flex_4', 'R_flex_5']].var(axis=1)
        df['L_flex_variance'] = df[['L_flex_1', 'L_flex_2', 'L_flex_3', 'L_flex_4', 'L_flex_5']].var(axis=1)
        
        # Hand orientation features
        df['R_roll'] = np.arctan2(df['R_accel_y'], df['R_accel_z']) * 180 / np.pi
        df['R_pitch'] = np.arctan2(-df['R_accel_x'], np.sqrt(df['R_accel_y']**2 + df['R_accel_z']**2)) * 180 / np.pi
        df['L_roll'] = np.arctan2(df['L_accel_y'], df['L_accel_z']) * 180 / np.pi
        df['L_pitch'] = np.arctan2(-df['L_accel_x'], np.sqrt(df['L_accel_y']**2 + df['L_accel_z']**2)) * 180 / np.pi
        
        return df
    
    def create_window_features(self, buffer_data, window_size=30):
        """
        Create statistical features from buffered sensor data
        """
        if len(buffer_data) < window_size:
            return None
        
        # Use last window_size samples
        window_df = pd.DataFrame(list(buffer_data)[-window_size:])
        
        features = {}
        
        # Calculate statistical features for each sensor
        for col in window_df.columns:
            if col in self.sensor_columns:
                features[f'{col}_mean'] = window_df[col].mean()
                features[f'{col}_std'] = window_df[col].std()
                features[f'{col}_min'] = window_df[col].min()
                features[f'{col}_max'] = window_df[col].max()
                features[f'{col}_range'] = window_df[col].max() - window_df[col].min()
                features[f'{col}_q25'] = window_df[col].quantile(0.25)
                features[f'{col}_q75'] = window_df[col].quantile(0.75)
                
                # Slope (trend)
                x = np.arange(len(window_df))
                if len(window_df) > 1:
                    slope, _, _, _, _ = stats.linregress(x, window_df[col])
                    features[f'{col}_slope'] = slope
                else:
                    features[f'{col}_slope'] = 0
        
        return pd.DataFrame([features])
    
    def predict_gesture(self, sensor_data):
        """
        Predict gesture from current sensor data
        """
        try:
            # Preprocess data
            processed_data = self.preprocess_sensor_data(sensor_data)
            
            # Add to buffer
            self.sensor_buffer.append(processed_data.iloc[0].to_dict())
            
            # Create windowed features if buffer is full enough
            window_features = self.create_window_features(self.sensor_buffer)
            
            if window_features is None:
                return self.last_prediction
            
            # Ensure all required features are present
            missing_features = set(self.feature_columns) - set(window_features.columns)
            for feature in missing_features:
                window_features[feature] = 0  # Fill missing features with 0
            
            # Select and order features correctly
            X = window_features[self.feature_columns]
            
            # Scale features
            X_scaled = self.scaler.transform(X)
            
            # Predict
            prediction = self.model.predict(X_scaled)
            confidence = self.model.predict_proba(X_scaled).max()
            
            # Decode label
            gesture_label = self.label_encoder.inverse_transform(prediction)[0]
            
            # Update prediction if confidence is high enough
            if confidence >= self.prediction_threshold:
                self.last_prediction = {
                    "gesture": gesture_label,
                    "confidence": float(confidence),
                    "timestamp": time.time()
                }
            
            return self.last_prediction
            
        except Exception as e:
            self.logger.error(f"Prediction error: {e}")
            return self.last_prediction
    
    def start_serial_communication(self, port='/dev/ttyUSB0', baudrate=115200):
        """
        Start serial communication with ESP32 devices
        """
        try:
            self.serial_connection = serial.Serial(port, baudrate, timeout=1)
            self.logger.info(f"Serial connection established on {port}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to establish serial connection: {e}")
            return False
    
    def read_sensor_data_serial(self):
        """
        Read sensor data from ESP32 via serial
        Expected format: JSON string with all sensor values
        """
        try:
            if hasattr(self, 'serial_connection') and self.serial_connection.is_open:
                line = self.serial_connection.readline().decode('utf-8').strip()
                if line:
                    sensor_data = json.loads(line)
                    return sensor_data
        except Exception as e:
            self.logger.error(f"Error reading serial data: {e}")
        return None
    
    def start_udp_server(self, host='0.0.0.0', port=8888):
        """
        Start UDP server to receive sensor data from ESP32 devices
        """
        try:
            self.udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.udp_socket.bind((host, port))
            self.logger.info(f"UDP server started on {host}:{port}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to start UDP server: {e}")
            return False
    
    def read_sensor_data_udp(self):
        """
        Read sensor data from ESP32 via UDP
        """
        try:
            if hasattr(self, 'udp_socket'):
                data, addr = self.udp_socket.recvfrom(1024)
                sensor_data = json.loads(data.decode('utf-8'))
                return sensor_data
        except Exception as e:
            self.logger.error(f"Error reading UDP data: {e}")
        return None
    
    def run_prediction_loop(self, communication_type='serial'):
        """
        Main prediction loop
        """
        self.running = True
        self.logger.info("Starting real-time ASL prediction...")
        
        while self.running:
            try:
                # Read sensor data based on communication type
                if communication_type == 'serial':
                    raw_data = self.read_sensor_data_serial()
                elif communication_type == 'udp':
                    raw_data = self.read_sensor_data_udp()
                else:
                    self.logger.error("Invalid communication type")
                    break
                
                if raw_data is not None:
                    # Convert to expected format (list of sensor values)
                    sensor_values = [raw_data.get(col, 0) for col in self.sensor_columns]
                    
                    # Predict gesture
                    prediction = self.predict_gesture(sensor_values)
                    
                    # Log prediction
                    self.logger.info(f"Gesture: {prediction['gesture']}, "
                                   f"Confidence: {prediction['confidence']:.3f}")
                    
                    # Put prediction in queue for other processes
                    if not self.prediction_queue.full():
                        self.prediction_queue.put(prediction)
                
                time.sleep(0.05)  # 20 Hz update rate
                
            except KeyboardInterrupt:
                self.logger.info("Stopping prediction loop...")
                break
            except Exception as e:
                self.logger.error(f"Error in prediction loop: {e}")
                time.sleep(0.1)
        
        self.running = False
    
    def start_async_prediction(self, communication_type='serial'):
        """
        Start prediction loop in separate thread
        """
        self.prediction_thread = threading.Thread(
            target=self.run_prediction_loop,
            args=(communication_type,)
        )
        self.prediction_thread.daemon = True
        self.prediction_thread.start()
    
    def stop_prediction(self):
        """
        Stop prediction loop
        """
        self.running = False
        if self.prediction_thread:
            self.prediction_thread.join()
        
        # Close connections
        if hasattr(self, 'serial_connection'):
            self