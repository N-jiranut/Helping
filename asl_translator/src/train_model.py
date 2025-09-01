import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime

# Add src directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from asl_ml_pipeline import ASLGestureRecognizer
from config import RAW_DATA_DIR, MODEL_DIR, LOG_DIR

def main():
    print("🚀 Starting ASL Model Training")
    print("=" * 50)
    
    # Check if data file exists
    data_file = os.path.join(RAW_DATA_DIR, 'RE_14-Aug.csv')
    if not os.path.exists(data_file):
        print(f"❌ Data file not found: {data_file}")
        print("Please place your CSV data file in the data/raw/ directory")
        print("Expected columns:")
        from config import SENSOR_COLUMNS
        print(f"timestamp,{','.join(SENSOR_COLUMNS)},result")
        return
    
    try:
        # Load data
        print(f"📁 Loading data from {data_file}")
        df = pd.read_csv(data_file)
        print(f"✅ Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        print(f"📊 Unique gestures: {df['result'].nunique()}")
        print(f"🏷️  Gestures: {list(df['result'].unique())}")
        
        # Initialize recognizer
        asl_recognizer = ASLGestureRecognizer()
        
        # Preprocess data
        print("\n🔄 Preprocessing data...")
        processed_df = asl_recognizer.preprocess_data(df)
        print(f"✅ Preprocessing complete: {processed_df.shape}")
        
        # Create windowed features
        print("🪟 Creating windowed features...")
        windowed_df = asl_recognizer.create_windowed_features(processed_df)
        print(f"✅ Windowed features created: {windowed_df.shape}")
        
        # Prepare features and labels
        X = windowed_df.drop(['result'], axis=1)
        y = windowed_df['result']
        
        print(f"\n📈 Training data prepared:")
        print(f"   Features: {X.shape}")
        print(f"   Labels: {len(y.unique())} unique gestures")
        
        # Train model
        print("\n🤖 Training model...")
        X_test, y_test, y_pred = asl_recognizer.train_model(X, y)
        
        # Save model
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = os.path.join(MODEL_DIR, f'asl_model_{timestamp}.pkl')
        asl_recognizer.save_model(model_path)
        
        # Also save as default model
        default_model_path = os.path.join(MODEL_DIR, 'asl_gesture_model.pkl')
        asl_recognizer.save_model(default_model_path)
        
        print(f"\n✅ Training complete!")
        print(f"📁 Model saved to: {model_path}")
        print(f"📁 Default model: {default_model_path}")
        
        # Generate plots (optional)
        try:
            print("\n📊 Generating visualizations...")
            asl_recognizer.plot_confusion_matrix(y_test, y_pred)
            asl_recognizer.plot_feature_importance(top_n=20)
        except Exception as e:
            print(f"⚠️  Could not generate plots: {e}")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()