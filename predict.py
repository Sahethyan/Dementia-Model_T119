"""
Dementia Prediction Script
Author:  Sahethyan,Yousuf,Ackaash
Purpose: Create test data and make predictions using a pre-trained model
"""

import pandas as pd
import numpy as np
import joblib
import json


def create_test_file():
    """
    Creates a new CSV file with test data for prediction.
    
    This function creates a multiline CSV string with sample patient data
    and writes it to new_prediction_data.csv.
    """
    # Define the CSV data as a multiline string
    csv_data = """BIRTHYR,VISITYR,AGE,SEX,EDUC,MARISTAT,NACCLIVS,INRELTO,AYRSPAN,APCENGL
1948,2015,67,1,18,1,2,1,0,100
1935,2016,81,2,12,2,1,2,0,100
1952,2017,65,1,16,1,2,1,10,90
1933,2015,82,2,10,2,1,3,0,100
1960,2018,58,1,20,5,1,5,0,100
1941,2019,78,2,14,2,3,2,0,100
1945,2017,72,1,16,1,2,1,0,100
1928,2016,88,2,8,2,4,2,20,80
1955,2018,63,1,17,1,2,1,0,100
1938,2020,82,2,12,2,1,2,0,100"""
    
    # Write the CSV data to file
    with open('new_prediction_data.csv', 'w') as f:
        f.write(csv_data)
    
    print("✓ Successfully created new_prediction_data.csv")


def run_predictions():
    """
    Loads a pre-trained model and makes predictions on new data.
    
    This function:
    1. Loads the saved scikit-learn pipeline from dementia_model.joblib
    2. Loads the new prediction data
    3. Makes class and probability predictions
    4. Saves results to final_predictions.csv
    """
    # Define file constants
    MODEL_FILE = 'dementia_model.joblib'
    NEW_DATA_FILE = 'new_prediction_data.csv'
    OUTPUT_FILE = 'final_predictions.csv'
    
    # Load Model
    print("\n--- Loading Pre-trained Model ---")
    try:
        pipeline = joblib.load(MODEL_FILE)
        print(f"✓ Successfully loaded model from {MODEL_FILE}")
    except FileNotFoundError:
        print(f"✗ ERROR: Model file '{MODEL_FILE}' not found.")
        print("   Please run the training script (ml_pipeline.py) first to create the model.")
        return
    except Exception as e:
        print(f"✗ ERROR: Failed to load model: {e}")
        return
    
    # Load New Data
    print("\n--- Loading New Prediction Data ---")
    try:
        df_new = pd.read_csv(NEW_DATA_FILE)
        print(f"✓ Successfully loaded data: {df_new.shape[0]} rows, {df_new.shape[1]} columns")
    except FileNotFoundError:
        print(f"✗ ERROR: Data file '{NEW_DATA_FILE}' not found.")
        print("   Please ensure new_prediction_data.csv exists.")
        return
    except Exception as e:
        print(f"✗ ERROR: Failed to load data: {e}")
        return
    
    # Get expected feature columns from the pipeline
    # The preprocessor in the pipeline knows which columns it expects
    print("\n--- Aligning Data with Model Features ---")
    try:
        # Get the preprocessor from the pipeline
        preprocessor = pipeline.named_steps['preprocessor']
        
        # Get all expected feature columns from the ColumnTransformer
        expected_features = []
        for name, transformer, features in preprocessor.transformers_:
            if name != 'remainder':
                expected_features.extend(features)
        
        print(f"✓ Model expects {len(expected_features)} features")
        
        # Create a DataFrame with all expected columns, filling missing ones with NaN
        # This allows the pipeline's imputers to handle missing values
        df_aligned = pd.DataFrame(index=df_new.index)
        
        for col in expected_features:
            if col in df_new.columns:
                df_aligned[col] = df_new[col]
            else:
                # Fill missing columns with NaN (will be handled by imputers)
                df_aligned[col] = np.nan
                print(f"  ⚠ Added missing column '{col}' with NaN values (will be imputed)")
        
        print(f"✓ Data aligned: {df_aligned.shape[0]} rows, {df_aligned.shape[1]} columns")
        
    except Exception as e:
        print(f"✗ ERROR: Failed to align data with model features: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Make Predictions
    print("\n--- Making Predictions ---")
    try:
        # --- UPDATED PREDICTION LOGIC ---
        
        # Load the optimal threshold we saved during training
        try:
            with open('model_config.json', 'r') as f:
                config = json.load(f)
                OPTIMAL_THRESHOLD = config['threshold']
            print(f"✓ Loaded optimal threshold: {OPTIMAL_THRESHOLD:.4f}")
        except FileNotFoundError:
            print("⚠ WARNING: 'model_config.json' not found. Using default threshold 0.5")
            print("  Please run the training script (ml_pipeline.py) first to generate the optimal threshold.")
            # Fallback to 0.5, though this will likely fail
            OPTIMAL_THRESHOLD = 0.5
        
        # 1. Get the probabilities (the "risk score")
        predictions_proba = pipeline.predict_proba(df_aligned)[:, 1]
        
        # 2. Make the class prediction (0 or 1) using our new, smarter threshold
        predictions_class = (predictions_proba >= OPTIMAL_THRESHOLD).astype(int)
        
        # --- END OF UPDATED LOGIC ---
        
        print(f"✓ Successfully generated predictions for {len(predictions_class)} samples")
        print(f"  Using threshold: {OPTIMAL_THRESHOLD:.4f}")
        print(f"  Predictions - Class 0: {(predictions_class == 0).sum()}, Class 1: {(predictions_class == 1).sum()}")
    except Exception as e:
        print(f"✗ ERROR: Prediction failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Format Output
    print("\n--- Formatting Output ---")
    
    # Create output DataFrame with original input data (not aligned data)
    df_output = df_new.copy()
    
    # Add prediction columns
    df_output['DEMENTIA_PREDICTION'] = predictions_class
    df_output['DEMENTIA_PROBABILITY'] = np.round(predictions_proba, 4)
    
    print("✓ Output DataFrame created with predictions")
    
    
    print("\n" + "=" * 80)
    print("--- Model Predictions ---")
    print("=" * 80)
    print(df_output.head())
    print("\n" + "=" * 80)
    

    try:
        df_output.to_csv(OUTPUT_FILE, index=False)
        print(f"\n✓ Predictions saved to {OUTPUT_FILE}")
    except Exception as e:
        print(f"✗ ERROR: Failed to save predictions: {e}")
        return
    
    print("\n" + "=" * 80)
    print("PREDICTION PROCESS COMPLETED SUCCESSFULLY")
    print("=" * 80)


if __name__ == "__main__":
    """
    Main execution block.
    First creates the test data file, then runs predictions.
    """
    
    print("=" * 80)
    print("CREATING TEST DATA FILE")
    print("=" * 80)
    create_test_file()
    
    # Step 2: Run predictions
    print("\n" + "=" * 80)
    print("RUNNING PREDICTIONS")
    print("=" * 80)
    run_predictions()

