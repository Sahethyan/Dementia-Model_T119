"""
Final Dementia Prediction Script
Author:  Sahethyan,Yousuf,Ackaash
Purpose: Make predictions using the trained model and optimal decision threshold
"""

import joblib
import pandas as pd
import numpy as np
import json


def run_predictions():
    """
    Main prediction function that:
    1. Loads the trained model from dementia_model.joblib
    2. Loads the optimal threshold from model_config.json
    3. Loads prediction data from mixed_prediction_data.csv
    4. Gets risk scores (probabilities) from the model
    5. Makes final predictions using the optimal threshold
    6. Saves results to mixed_predictions_output.csv
    7. Prints the full 50-row result DataFrame
    """
    # Define file paths
    MODEL_FILE = 'dementia_model.joblib'
    CONFIG_FILE = 'model_config.json'
    DATA_FILE = 'mixed_prediction_data.csv'
    OUTPUT_FILE = 'mixed_predictions_output.csv'
    
    # Step 1: Load the trained model
    print("\n--- Loading Trained Model ---")
    try:
        pipeline = joblib.load(MODEL_FILE)
        print(f"✓ Successfully loaded model from {MODEL_FILE}")
    except FileNotFoundError:
        print(f"✗ ERROR: Model file '{MODEL_FILE}' not found.")
        return
    except Exception as e:
        print(f"✗ ERROR: Failed to load model: {e}")
        return
    
    # Step 2: Load the optimal threshold from config
    print("\n--- Loading Optimal Threshold ---")
    try:
        with open(CONFIG_FILE, 'r') as f:
            config = json.load(f)
            OPTIMAL_THRESHOLD = config['threshold']
        print(f"✓ Loaded optimal threshold: {OPTIMAL_THRESHOLD:.4f}")
    except FileNotFoundError:
        print(f"✗ ERROR: Config file '{CONFIG_FILE}' not found.")
        return
    except KeyError:
        print(f"✗ ERROR: 'threshold' key not found in {CONFIG_FILE}.")
        return
    except Exception as e:
        print(f"✗ ERROR: Failed to load config: {e}")
        return
    
    # Step 3: Load the prediction data
    print("\n--- Loading Prediction Data ---")
    try:
        df_data = pd.read_csv(DATA_FILE)
        print(f"✓ Successfully loaded data: {df_data.shape[0]} rows, {df_data.shape[1]} columns")
    except FileNotFoundError:
        print(f"✗ ERROR: Data file '{DATA_FILE}' not found.")
        return
    except Exception as e:
        print(f"✗ ERROR: Failed to load data: {e}")
        return
    
    # Step 4: Align data with model features
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
        df_aligned = pd.DataFrame(index=df_data.index)
        
        for col in expected_features:
            if col in df_data.columns:
                df_aligned[col] = df_data[col]
            else:
                # Fill missing columns with NaN (will be handled by imputers)
                df_aligned[col] = np.nan
        
        print(f"✓ Data aligned: {df_aligned.shape[0]} rows, {df_aligned.shape[1]} columns")
        
    except Exception as e:
        print(f"✗ ERROR: Failed to align data with model features: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Step 5: Get probabilities (risk scores) from the model
    print("\n--- Generating Risk Scores (Probabilities) ---")
    try:
        # Get the probabilities for class 1 (dementia risk)
        risk_scores = pipeline.predict_proba(df_aligned)[:, 1]
        print(f"✓ Successfully generated risk scores for {len(risk_scores)} samples")
        print(f"  Risk score range: {risk_scores.min():.4f} - {risk_scores.max():.4f}")
    except Exception as e:
        print(f"✗ ERROR: Failed to generate risk scores: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Step 6: Make final predictions using the optimal threshold
    print("\n--- Making Final Predictions with Optimal Threshold ---")
    try:
        # Use the OPTIMAL_THRESHOLD to make binary predictions (0 or 1)
        dementia_predictions = (risk_scores >= OPTIMAL_THRESHOLD).astype(int)
        
        print(f"✓ Successfully generated predictions using threshold: {OPTIMAL_THRESHOLD:.4f}")
        print(f"  Predictions - Class 0 (No Dementia): {(dementia_predictions == 0).sum()}")
        print(f"  Predictions - Class 1 (Dementia): {(dementia_predictions == 1).sum()}")
    except Exception as e:
        print(f"✗ ERROR: Failed to make predictions: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Step 7: Create output DataFrame
    print("\n--- Formatting Results ---")
    df_results = df_data.copy()
    df_results['DEMENTIA_PREDICTION'] = dementia_predictions
    df_results['DEMENTIA_PROBABILITY'] = np.round(risk_scores, 4)
    
    print("✓ Results DataFrame created")
    
    # Step 8: Save results to CSV
    print("\n--- Saving Results ---")
    try:
        df_results.to_csv(OUTPUT_FILE, index=False)
        print(f"✓ Results saved to {OUTPUT_FILE}")
    except Exception as e:
        print(f"✗ ERROR: Failed to save results: {e}")
        return
    
    # Step 9: Print the full 50-row result DataFrame
    print("\n" + "=" * 100)
    print("FULL PREDICTION RESULTS (50 ROWS)")
    print("=" * 100)
    
    # Configure pandas to display all rows and columns
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)
    
    print(df_results)
    print("=" * 100)
    
    print("\n✓ Prediction process completed successfully!")
    print(f"✓ Total predictions: {len(df_results)} rows")
    print(f"✓ Output file: {OUTPUT_FILE}")


if __name__ == "__main__":
    """
    Main execution block.
    """
    print("=" * 100)
    print("FINAL DEMENTIA PREDICTION SCRIPT")
    print("=" * 100)
    run_predictions()

