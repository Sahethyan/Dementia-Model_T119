"""
Comprehensive Train-and-Predict Workflow for Dementia Prediction Model
Author:  Sahethyan,Yousuf,Ackaash
Purpose: Re-train model with class imbalance fix and immediately verify with predictions
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from lightgbm import LGBMClassifier
from sklearn.metrics import f1_score, roc_auc_score


def train_and_save_model():
    """
    Re-train the dementia prediction model with class imbalance fix and save it.
    
    This function:
    1. Loads the training data
    2. Splits into train/test sets
    3. Calculates scale_pos_weight to fix class imbalance
    4. Builds preprocessing pipeline
    5. Performs hyperparameter tuning with RandomizedSearchCV
    6. Saves the best tuned model
    """
    print("=" * 80)
    print("TRAINING NEW MODEL WITH CLASS IMBALANCE FIX")
    print("=" * 80)
    
    # Constants
    TRAIN_DATA_FILE = 'cleaned_Dementia_Prediction_Dataset.csv'
    MODEL_OUTPUT_FILE = 'dementia_model.joblib'
    
    # Load Data
    print(f"\n--- Step 1: Loading Training Data ---")
    try:
        df = pd.read_csv(TRAIN_DATA_FILE)
        print(f"✓ Successfully loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    except FileNotFoundError:
        print(f"✗ ERROR: File '{TRAIN_DATA_FILE}' not found.")
        return None
    except Exception as e:
        print(f"✗ ERROR: Failed to load data: {e}")
        return None
    
    # Define Features & Target
    print(f"\n--- Step 2: Defining Features and Target ---")
    TARGET = 'DEMENTED'
    
    # Numerical features
    numerical_features = [
        'BIRTHMO', 'BIRTHYR', 'EDUC', 'NACCAGE', 'NACCAGEB', 'INBIRMO', 'INBIRYR', 
        'INEDUC', 'INKNOWN', 'AYRSPAN', 'AYRENGL', 'APCSPAN', 'APCENGL', 'NACCSPNL', 
        'NACCENGL', 'VISITMO', 'VISITDAY', 'VISITYR', 'NACCVNUM', 'NACCAVST', 
        'NACCNVST', 'NACCDAYS', 'NACCFDYS'
    ]
    
    # Categorical features
    categorical_features = [
        'NACCREAS', 'NACCREFR', 'SEX', 'HISPANIC', 'HISPOR', 'RACE', 'RACESEC', 
        'RACETER', 'PRIMLANG', 'MARISTAT', 'NACCLIVS', 'INDEPEND', 'RESIDENC', 
        'HANDED', 'NACCNIHR', 'INSEX', 'NEWINF', 'INHISP', 'INHISPOR', 'INRACE', 
        'INRASEC', 'INRATER', 'INRELTO', 'INLIVWTH', 'INVISITS', 'INCALLS', 
        'INRELY', 'NACCNINR', 'APREFLAN', 'ASPKSPAN', 'AREASPAN', 'AWRISPAN', 
        'AUNDSPAN', 'ASPKENGL', 'AREAENGL', 'AWRIENGL', 'AUNDENGL'
    ]
    
    # Verify target exists
    if TARGET not in df.columns:
        print(f"✗ ERROR: Target variable '{TARGET}' not found in dataset.")
        return None
    
    # Filter features to only those that exist in the dataset
    numerical_features = [f for f in numerical_features if f in df.columns]
    categorical_features = [f for f in categorical_features if f in df.columns]
    
    print(f"✓ Using {len(numerical_features)} numerical features")
    print(f"✓ Using {len(categorical_features)} categorical features")
    
    # Prepare X and y
    all_feature_cols = numerical_features + categorical_features
    X = df[all_feature_cols].copy()
    y = df[TARGET].copy()
    
    print(f"✓ Feature matrix X shape: {X.shape}")
    print(f"✓ Target vector y shape: {y.shape}")
    print(f"✓ Target distribution:\n{y.value_counts()}")
    
    # Split Data
    print(f"\n--- Step 3: Train-Test Split ---")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y  # Critical for imbalanced dataset
    )
    
    print(f"✓ Training set: {X_train.shape[0]} samples")
    print(f"✓ Test set: {X_test.shape[0]} samples")
    print(f"✓ Training target distribution:\n{y_train.value_counts()}")
    
    # Fix Imbalance: Calculate scale_pos_weight
    print(f"\n--- Step 4: Calculating Class Imbalance Fix ---")
    scale_pos_weight = y_train.value_counts()[0] / y_train.value_counts()[1]
    print(f"✓ Class Imbalance Ratio (scale_pos_weight): {scale_pos_weight:.2f}")
    print(f"  (Majority class: {y_train.value_counts()[0]}, Minority class: {y_train.value_counts()[1]})")
    
    # Build Preprocessor
    print(f"\n--- Step 5: Building Preprocessing Pipeline ---")
    
    # Numerical transformer: median imputation + standardization
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Categorical transformer: mode imputation + one-hot encoding
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    # Column transformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='drop'
    )
    
    print("✓ Preprocessing pipeline created")
    
    # Build Tunable Pipeline
    print(f"\n--- Step 6: Building Model Pipeline ---")
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LGBMClassifier(random_state=42, verbose=-1))
    ])
    
    print("✓ Model pipeline created")
    
    # Tune the Model (Hyperparameter Tuning)
    print(f"\n--- Step 7: Hyperparameter Tuning ---")
    print("This may take several minutes...")
    
    # Define parameter distribution
    # CRITICAL: Include scale_pos_weight in the grid
    param_dist = {
        'classifier__n_estimators': [100, 200, 500],
        'classifier__learning_rate': [0.01, 0.05, 0.1],
        'classifier__num_leaves': [20, 31, 50],
        'classifier__scale_pos_weight': [scale_pos_weight]  # Fix for class imbalance
    }
    
    # Create RandomizedSearchCV
    random_search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_dist,
        n_iter=25,
        cv=3,
        scoring='f1',
        random_state=42,
        n_jobs=-1,
        verbose=1
    )
    
    # Run hyperparameter tuning
    try:
        random_search.fit(X_train, y_train)
        print("✓ Hyperparameter tuning completed successfully")
        print(f"✓ Best F1 score (CV): {random_search.best_score_:.4f}")
        print(f"✓ Best parameters: {random_search.best_params_}")
        
        # Evaluate on test set
        tuned_model = random_search.best_estimator_
        y_pred = tuned_model.predict(X_test)
        y_proba = tuned_model.predict_proba(X_test)[:, 1]
        
        test_f1 = f1_score(y_test, y_pred)
        test_roc_auc = roc_auc_score(y_test, y_proba)
        
        print(f"\n--- Test Set Performance ---")
        print(f"✓ Test F1 Score: {test_f1:.4f}")
        print(f"✓ Test ROC AUC: {test_roc_auc:.4f}")
        
    except Exception as e:
        print(f"✗ ERROR: Hyperparameter tuning failed: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # Save the Best Model
    print(f"\n--- Step 8: Saving Best Model ---")
    tuned_model = random_search.best_estimator_
    
    try:
        joblib.dump(tuned_model, MODEL_OUTPUT_FILE)
        print(f"✓ Model saved successfully to {MODEL_OUTPUT_FILE}")
        print(f"\n--- New, smarter model saved to {MODEL_OUTPUT_FILE} ---")
    except Exception as e:
        print(f"✗ ERROR: Failed to save model: {e}")
        return None
    
    print("\n" + "=" * 80)
    print("MODEL TRAINING COMPLETED SUCCESSFULLY")
    print("=" * 80)
    
    return tuned_model


def run_predictions():
    """
    Load the newly trained model and run predictions on mixed_prediction_data.csv.
    
    This function:
    1. Loads the saved model
    2. Loads the new prediction data
    3. Makes predictions (both class and probability)
    4. Saves results to CSV
    5. Displays results
    """
    print("\n" + "=" * 80)
    print("RUNNING PREDICTIONS WITH NEW MODEL")
    print("=" * 80)
    
    # Constants
    MODEL_FILE = 'dementia_model.joblib'
    NEW_DATA_FILE = 'mixed_prediction_data.csv'
    PREDICTION_OUTPUT_FILE = 'mixed_predictions_output.csv'
    
    # Load Model
    print(f"\n--- Step 1: Loading Trained Model ---")
    try:
        pipeline = joblib.load(MODEL_FILE)
        print(f"✓ Successfully loaded model from {MODEL_FILE}")
    except FileNotFoundError:
        print(f"✗ ERROR: Model file '{MODEL_FILE}' not found.")
        print("  Please run train_and_save_model() first.")
        return
    except Exception as e:
        print(f"✗ ERROR: Failed to load model: {e}")
        return
    
    # Load New Data
    print(f"\n--- Step 2: Loading Prediction Data ---")
    try:
        df_new = pd.read_csv(NEW_DATA_FILE)
        print(f"✓ Successfully loaded prediction data: {df_new.shape[0]} rows, {df_new.shape[1]} columns")
    except FileNotFoundError:
        print(f"✗ ERROR: File '{NEW_DATA_FILE}' not found.")
        return
    except Exception as e:
        print(f"✗ ERROR: Failed to load prediction data: {e}")
        return
    
    # Align Data with Model Features
    print(f"\n--- Step 3: Aligning Data with Model Features ---")
    try:
        # Get the preprocessor from the pipeline
        preprocessor = pipeline.named_steps['preprocessor']
        
        # Get all expected feature columns from the ColumnTransformer
        expected_features = []
        for name, transformer, features in preprocessor.transformers_:
            if name != 'remainder':
                expected_features.extend(features)
        
        print(f"✓ Model expects {len(expected_features)} features")
        print(f"✓ Input data has {len(df_new.columns)} features")
        
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
        print(f"  (Missing columns filled with NaN for imputation)")
        
    except Exception as e:
        print(f"✗ ERROR: Failed to align data with model features: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Make Predictions
    print(f"\n--- Step 4: Making Predictions ---")
    try:
        predictions_class = pipeline.predict(df_aligned)
        predictions_proba = pipeline.predict_proba(df_aligned)[:, 1]
        print(f"✓ Successfully generated predictions for {len(df_aligned)} samples")
    except Exception as e:
        print(f"✗ ERROR: Prediction failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Format & Save Output
    print(f"\n--- Step 5: Formatting and Saving Results ---")
    
    # Create predictions DataFrame
    predictions_df = df_new.copy()
    predictions_df['DEMENTIA_PREDICTION'] = predictions_class
    predictions_df['DEMENTIA_PROBABILITY'] = predictions_proba
    
    # Save to CSV
    try:
        predictions_df.to_csv(PREDICTION_OUTPUT_FILE, index=False)
        print(f"✓ Predictions saved to {PREDICTION_OUTPUT_FILE}")
    except Exception as e:
        print(f"✗ ERROR: Failed to save predictions: {e}")
        return
    
    # Display Results
    print(f"\n--- Predictions from NEW Model ---")
    print("=" * 80)
    
    # Print summary statistics
    print(f"\nPrediction Summary:")
    print(f"  Total samples: {len(predictions_df)}")
    print(f"  Predicted Class 0 (No Dementia): {(predictions_df['DEMENTIA_PREDICTION'] == 0).sum()}")
    print(f"  Predicted Class 1 (Dementia): {(predictions_df['DEMENTIA_PREDICTION'] == 1).sum()}")
    print(f"  Average probability: {predictions_df['DEMENTIA_PROBABILITY'].mean():.4f}")
    print(f"  Min probability: {predictions_df['DEMENTIA_PROBABILITY'].min():.4f}")
    print(f"  Max probability: {predictions_df['DEMENTIA_PROBABILITY'].max():.4f}")
    
    # Print all predictions
    print(f"\n--- All Predictions (Full DataFrame) ---")
    print(predictions_df.to_string())
    
    print("\n" + "=" * 80)
    print("PREDICTIONS COMPLETED SUCCESSFULLY")
    print("=" * 80)


if __name__ == "__main__":
    """
    Main execution block: Train model first, then run predictions.
    """
    print("\n" + "=" * 80)
    print("COMPREHENSIVE TRAIN-AND-PREDICT WORKFLOW")
    print("=" * 80)
    
    # Step 1: Train and save the new model
    trained_model = train_and_save_model()
    
    # Step 2: Run predictions with the new model
    if trained_model is not None:
        run_predictions()
    else:
        print("\n⚠ WARNING: Model training failed. Skipping predictions.")
    
    print("\n" + "=" * 80)
    print("WORKFLOW COMPLETED")
    print("=" * 80)

