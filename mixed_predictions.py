"""
Dementia Prediction Script for Mixed Dataset
Author: Senior ML Engineer
Purpose: Load a pre-trained scikit-learn pipeline and make predictions on new, unseen data
"""

import pandas as pd
import numpy as np
import joblib


def main():
    """
    Main function to load model and make predictions on mixed prediction data.
    
    This function:
    1. Loads the trained pipeline from dementia_model.joblib
    2. Loads the new prediction data from mixed_prediction_data.csv
    3. Makes class and probability predictions
    4. Saves results to mixed_predictions_output.csv
    """
    # Define Constants
    MODEL_FILE = 'dementia_model.joblib'  # This is the file where our trained pipeline was saved
    NEW_DATA_FILE = 'mixed_prediction_data.csv'  # This is the new 50-row dataset
    OUTPUT_FILE = 'mixed_predictions_output.csv'  # Output file for predictions
    
    # Load Model
    print("\n--- Loading Pre-trained Model ---")
    try:
        pipeline = joblib.load(MODEL_FILE)
        print(f"✓ Successfully loaded model from {MODEL_FILE}")
    except FileNotFoundError:
        print(f"✗ ERROR: Model file '{MODEL_FILE}' not found.")
        print("   Please run the training script (ml_pipeline.py) first to create this file.")
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
        return
    except Exception as e:
        print(f"✗ ERROR: Failed to load data: {e}")
        return
    
    # Align Data with Model Features
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
        
        print(f"✓ Data aligned: {df_aligned.shape[0]} rows, {df_aligned.shape[1]} columns")
        
    except Exception as e:
        print(f"✗ ERROR: Failed to align data with model features: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Make Predictions
    print("\n--- Making Predictions ---")
    try:
        # Get class predictions (0 or 1)
        predictions_class = pipeline.predict(df_aligned)
        
        # Get probability predictions (probability of class 1)
        predictions_proba = pipeline.predict_proba(df_aligned)[:, 1]
        
        print(f"✓ Successfully generated predictions for {len(predictions_class)} samples")
    except Exception as e:
        print(f"✗ ERROR: Prediction failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Format Output
    print("\n--- Formatting Output ---")
    
    # Create a new DataFrame called predictions_df
    # Copy the original data from df_new into it
    predictions_df = df_new.copy()
    
    # Add two new columns:
    # 'DEMENTIA_PREDICTION' (with the predictions_class data)
    predictions_df['DEMENTIA_PREDICTION'] = predictions_class
    
    # 'DEMENTIA_PROBABILITY' (with the predictions_proba data, rounded to 4 decimal places)
    predictions_df['DEMENTIA_PROBABILITY'] = np.round(predictions_proba, 4)
    
    print("✓ Output DataFrame created with predictions")
    
    # Display and Save
    print("\n" + "=" * 80)
    print("--- Model Predictions on Mixed Dataset ---")
    print("=" * 80)
    # Print all 50 rows, not just the head
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)
    print(predictions_df)
    print("=" * 80)
    
    # Save the predictions_df to OUTPUT_FILE, with index=False
    try:
        predictions_df.to_csv(OUTPUT_FILE, index=False)
        print(f"\n✓ Predictions saved to {OUTPUT_FILE}")
    except Exception as e:
        print(f"✗ ERROR: Failed to save predictions: {e}")
        return
    
    # Print a final confirmation that the file was saved
    print(f"\n✓ File saved successfully: {OUTPUT_FILE}")
    print(f"✓ Total predictions: {len(predictions_df)} rows")


if __name__ == "__main__":
    """
    Main execution block.
    """
    main()

