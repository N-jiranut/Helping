import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import butter, filtfilt
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class ASLGestureRecognizer:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.model = None
        self.feature_columns = []
        
    def preprocess_data(self, df, apply_filters=True):
        """
        Preprocess sensor data with filtering and feature engineering
        """
        print("Preprocessing sensor data...")
        
        # Create a copy to avoid modifying original data
        processed_df = df.copy()
        
        # Define sensor columns
        sensor_columns = [
            'R_accel_x', 'R_accel_y', 'R_accel_z',
            'R_gyro_x', 'R_gyro_y', 'R_gyro_z',
            'R_flex_1', 'R_flex_2', 'R_flex_3', 'R_flex_4', 'R_flex_5',
            'L_accel_x', 'L_accel_y', 'L_accel_z',
            'L_gyro_x', 'L_gyro_y', 'L_gyro_z',
            'L_flex_1', 'L_flex_2', 'L_flex_3', 'L_flex_4', 'L_flex_5'
        ]
        
        if apply_filters:
            # Apply low-pass filter to reduce noise (Butterworth filter)
            processed_df = self._apply_butterworth_filter(processed_df, sensor_columns)
        
        # Create engineered features
        processed_df = self._create_features(processed_df)
        
        return processed_df
    
    def _apply_butterworth_filter(self, df, sensor_columns, cutoff=10, fs=50, order=4):
        """
        Apply Butterworth low-pass filter to sensor data
        """
        nyquist = 0.5 * fs
        normal_cutoff = cutoff / nyquist
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        
        filtered_df = df.copy()
        
        for col in sensor_columns:
            if col in df.columns:
                filtered_df[col] = filtfilt(b, a, df[col])
        
        return filtered_df
    
    def _create_features(self, df):
        """
        Create engineered features for better gesture recognition
        """
        # Calculate magnitude for accelerometer and gyroscope
        df['R_accel_magnitude'] = np.sqrt(df['R_accel_x']**2 + df['R_accel_y']**2 + df['R_accel_z']**2)
        df['L_accel_magnitude'] = np.sqrt(df['L_accel_x']**2 + df['L_accel_y']**2 + df['L_accel_z']**2)
        df['R_gyro_magnitude'] = np.sqrt(df['R_gyro_x']**2 + df['R_gyro_y']**2 + df['R_gyro_z']**2)
        df['L_gyro_magnitude'] = np.sqrt(df['L_gyro_x']**2 + df['L_gyro_y']**2 + df['L_gyro_z']**2)
        
        # Calculate total flex sensor values for each hand
        df['R_flex_total'] = df[['R_flex_1', 'R_flex_2', 'R_flex_3', 'R_flex_4', 'R_flex_5']].sum(axis=1)
        df['L_flex_total'] = df[['L_flex_1', 'L_flex_2', 'L_flex_3', 'L_flex_4', 'L_flex_5']].sum(axis=1)
        
        # Calculate flex sensor variance (finger position spread)
        df['R_flex_variance'] = df[['R_flex_1', 'R_flex_2', 'R_flex_3', 'R_flex_4', 'R_flex_5']].var(axis=1)
        df['L_flex_variance'] = df[['L_flex_1', 'L_flex_2', 'L_flex_3', 'L_flex_4', 'L_flex_5']].var(axis=1)
        
        # Hand orientation features (roll, pitch based on accelerometer)
        df['R_roll'] = np.arctan2(df['R_accel_y'], df['R_accel_z']) * 180 / np.pi
        df['R_pitch'] = np.arctan2(-df['R_accel_x'], np.sqrt(df['R_accel_y']**2 + df['R_accel_z']**2)) * 180 / np.pi
        df['L_roll'] = np.arctan2(df['L_accel_y'], df['L_accel_z']) * 180 / np.pi
        df['L_pitch'] = np.arctan2(-df['L_accel_x'], np.sqrt(df['L_accel_y']**2 + df['L_accel_z']**2)) * 180 / np.pi
        
        return df
    
    def create_windowed_features(self, df, window_size=30, step_size=15):
        """
        Create sliding window features for time-series data
        Useful for capturing gesture dynamics over time
        """
        print(f"Creating windowed features (window_size={window_size}, step_size={step_size})...")
        
        # Group by result label to process each gesture separately
        windowed_data = []
        
        for label in df['result'].unique():
            label_data = df[df['result'] == label].reset_index(drop=True)
            
            # Create sliding windows
            for i in range(0, len(label_data) - window_size + 1, step_size):
                window = label_data.iloc[i:i+window_size]
                
                # Extract statistical features from the window
                window_features = self._extract_window_features(window)
                window_features['result'] = label
                windowed_data.append(window_features)
        
        return pd.DataFrame(windowed_data)
    
    def _extract_window_features(self, window):
        """
        Extract statistical features from a time window
        """
        features = {}
        
        # Define feature columns (excluding timestamp and result)
        feature_cols = [col for col in window.columns if col not in ['timestamp', 'result']]
        
        for col in feature_cols:
            if col in window.columns:
                # Statistical features
                features[f'{col}_mean'] = window[col].mean()
                features[f'{col}_std'] = window[col].std()
                features[f'{col}_min'] = window[col].min()
                features[f'{col}_max'] = window[col].max()
                features[f'{col}_range'] = window[col].max() - window[col].min()
                
                # Percentiles
                features[f'{col}_q25'] = window[col].quantile(0.25)
                features[f'{col}_q75'] = window[col].quantile(0.75)
                
                # Slope (trend)
                x = np.arange(len(window))
                if len(window) > 1:
                    slope, _, _, _, _ = stats.linregress(x, window[col])
                    features[f'{col}_slope'] = slope
                else:
                    features[f'{col}_slope'] = 0
        
        return features
    
    def train_model(self, X, y, model_type='random_forest'):
        """
        Train the gesture recognition model
        """
        print("Training ASL gesture recognition model...")
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Store feature columns
        self.feature_columns = X.columns.tolist()
        
        # Initialize model based on type
        if model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1  # Use all CPU cores
            )
        
        # Train model
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate model
        y_pred = self.model.predict(X_test_scaled)
        
        print(f"\nModel Performance:")
        print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
        
        # Detailed classification report
        print("\nClassification Report:")
        print(classification_report(
            y_test, y_pred, 
            target_names=self.label_encoder.classes_
        ))
        
        # Cross-validation score
        cv_scores = cross_val_score(self.model, X_train_scaled, y_train, cv=5)
        print(f"Cross-validation accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        return X_test_scaled, y_test, y_pred
    
    def plot_feature_importance(self, top_n=20):
        """
        Plot feature importance for understanding model decisions
        """
        if self.model is None:
            print("Model not trained yet!")
            return
        
        # Get feature importances
        importances = self.model.feature_importances_
        feature_names = self.feature_columns
        
        # Create DataFrame and sort
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        # Plot top N features
        plt.figure(figsize=(12, 8))
        plt.barh(range(top_n), importance_df['importance'][:top_n][::-1])
        plt.yticks(range(top_n), importance_df['feature'][:top_n][::-1])
        plt.xlabel('Feature Importance')
        plt.title(f'Top {top_n} Most Important Features for ASL Gesture Recognition')
        plt.tight_layout()
        plt.show()
        
        return importance_df
    
    def plot_confusion_matrix(self, y_true, y_pred):
        """
        Plot confusion matrix
        """
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.label_encoder.classes_,
                    yticklabels=self.label_encoder.classes_)
        plt.title('Confusion Matrix - ASL Gesture Recognition')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.show()
    
    def predict_gesture(self, sensor_data):
        """
        Predict gesture from real-time sensor data
        Input: DataFrame with sensor readings or numpy array
        Output: Predicted gesture label and confidence
        """
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        # Convert to DataFrame if numpy array
        if isinstance(sensor_data, np.ndarray):
            sensor_data = pd.DataFrame(sensor_data, columns=self.feature_columns)
        
        # Ensure all required features are present
        missing_features = set(self.feature_columns) - set(sensor_data.columns)
        if missing_features:
            raise ValueError(f"Missing features: {missing_features}")
        
        # Scale features
        sensor_data_scaled = self.scaler.transform(sensor_data[self.feature_columns])
        
        # Predict
        prediction = self.model.predict(sensor_data_scaled)
        confidence = self.model.predict_proba(sensor_data_scaled).max(axis=1)
        
        # Decode label
        gesture_label = self.label_encoder.inverse_transform(prediction)
        
        return gesture_label, confidence
    
    def save_model(self, filepath):
        """
        Save trained model and preprocessors
        """
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'feature_columns': self.feature_columns
        }
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """
        Load trained model and preprocessors
        """
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.label_encoder = model_data['label_encoder']
        self.feature_columns = model_data['feature_columns']
        print(f"Model loaded from {filepath}")

# Example usage and training pipeline
def main():
    # Initialize the ASL recognizer
    asl_recognizer = ASLGestureRecognizer()
    
    # Load your data (replace with your actual data file)
    print("Loading data...")
    # df = pd.read_csv('your_asl_data.csv')
    
    # For demonstration, creating sample data structure
    # Replace this with your actual data loading
    sample_columns = [
        'timestamp', 'R_accel_x', 'R_accel_y', 'R_accel_z',
        'R_gyro_x', 'R_gyro_y', 'R_gyro_z', 'R_temp',
        'R_flex_1', 'R_flex_2', 'R_flex_3', 'R_flex_4', 'R_flex_5',
        'L_accel_x', 'L_accel_y', 'L_accel_z',
        'L_gyro_x', 'L_gyro_y', 'L_gyro_z', 'L_temp',
        'L_flex_1', 'L_flex_2', 'L_flex_3', 'L_flex_4', 'L_flex_5',
        'result'
    ]
    
    print("Expected data columns:", sample_columns)
    print("\nTo use this pipeline:")
    print("1. Load your CSV data with pd.read_csv()")
    print("2. Ensure your data has the columns listed above")
    print("3. Run the preprocessing and training steps")
    print("4. Save the trained model for real-time use on Raspberry Pi")
    
    # Example training workflow (uncomment when you have data):
    """
    # Preprocess data
    processed_df = asl_recognizer.preprocess_data(df)
    
    # Create windowed features for better gesture recognition
    windowed_df = asl_recognizer.create_windowed_features(processed_df)
    
    # Prepare features and labels
    X = windowed_df.drop(['result'], axis=1)
    y = windowed_df['result']
    
    # Train model
    X_test, y_test, y_pred = asl_recognizer.train_model(X, y)
    
    # Visualize results
    asl_recognizer.plot_confusion_matrix(y_test, y_pred)
    importance_df = asl_recognizer.plot_feature_importance()
    
    # Save model for Raspberry Pi deployment
    asl_recognizer.save_model('asl_gesture_model.pkl')
    
    # Example real-time prediction
    # new_sensor_data = pd.DataFrame(...) # Your real-time sensor readings
    # gesture, confidence = asl_recognizer.predict_gesture(new_sensor_data)
    # print(f"Predicted gesture: {gesture[0]} (confidence: {confidence[0]:.2f})")
    """

if __name__ == "__main__":
    main()