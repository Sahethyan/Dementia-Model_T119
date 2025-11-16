"""
ModelX Hackathon - Complete Training Script
Author: Senior ML Engineer
Purpose: End-to-end training workflow for dementia prediction model
"""

import pandas as pd
import numpy as np
import joblib
import json
import warnings
import os
from contextlib import contextmanager

# Data Visualization
import matplotlib.pyplot as plt

# Preprocessing & Pipelining
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Model
from lightgbm import LGBMClassifier

# Evaluation Metrics
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    precision_recall_curve, ConfusionMatrixDisplay, classification_report,
    confusion_matrix
)

# Explainability
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    print("Warning: SHAP library not found. Explainability plot will be skipped.")
    SHAP_AVAILABLE = False

# Utility to suppress warnings
warnings.filterwarnings('ignore')


@contextmanager
def suppress_stdout_stderr():
    """A context manager that redirects stdout and stderr to devnull"""
    import sys
    with open(os.devnull, 'w') as fnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        try:
            sys.stdout = fnull
            sys.stderr = fnull
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr


# ============================================================================
# 1. DEFINE CONSTANTS
# ============================================================================
CLEAN_DATA_FILE = 'cleaned_Dementia_Prediction_Dataset.csv'
MODEL_OUTPUT_FILE = 'dementia_model.joblib'
CONFIG_OUTPUT_FILE = 'model_config.json'
TARGET_VARIABLE = 'DEMENTED'

# Based on our analysis of "allowed" non-medical features
NUMERICAL_FEATURES = [
    'BIRTHMO', 'BIRTHYR', 'EDUC', 'NACCAGE', 'NACCAGEB', 'INBIRMO', 'INBIRYR', 
    'INEDUC', 'INKNOWN', 'AYRSPAN', 'AYRENGL', 'APCSPAN', 'APCENGL', 'NACCSPNL', 
    'NACCENGL', 'VISITMO', 'VISITDAY', 'VISITYR', 'NACCVNUM', 'NACCAVST', 
    'NACCNVST', 'NACCDAYS', 'NACCFDYS'
]

CATEGORICAL_FEATURES = [
    'NACCREAS', 'NACCREFR', 'SEX', 'HISPANIC', 'HISPOR', 'RACE', 'RACESEC', 
    'RACETER', 'PRIMLANG', 'MARISTAT', 'NACCLIVS', 'INDEPEND', 'RESIDENC', 
    'HANDED', 'NACCNIHR', 'INSEX', 'NEWINF', 'INHISP', 'INHISPOR', 'INRACE', 
    'INRASEC', 'INRATER', 'INRELTO', 'INLIVWTH', 'INVISITS', 'INCALLS', 
    'INRELY', 'NACCNINR', 'APREFLAN', 'ASPKSPAN', 'AREASPAN', 'AWRISPAN', 
    'AUNDSPAN', 'ASPKENGL', 'AREAENGL', 'AWRIENGL', 'AUNDENGL'
]


def main():
    """
    Main function to execute the complete training workflow.
    """
    print("=" * 80)
    print("MODELX HACKATHON - COMPLETE TRAINING WORKFLOW")
    print("=" * 80)
    
    # ============================================================================
    # 2. LOAD DATA
    # ============================================================================
    print("\n--- Step 1: Loading Data ---")
    try:
        df = pd.read_csv(CLEAN_DATA_FILE)
        print(f"✓ Successfully loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    except FileNotFoundError:
        print(f"✗ ERROR: File '{CLEAN_DATA_FILE}' not found.")
        return
    except Exception as e:
        print(f"✗ ERROR: Failed to load data: {e}")
        return
    
    # Verify target variable exists
    if TARGET_VARIABLE not in df.columns:
        print(f"✗ ERROR: Target variable '{TARGET_VARIABLE}' not found in dataset.")
        return
    
    # Filter features to only those that exist in the dataset
    numerical_features = [f for f in NUMERICAL_FEATURES if f in df.columns]
    categorical_features = [f for f in CATEGORICAL_FEATURES if f in df.columns]
    
    print(f"✓ Using {len(numerical_features)} numerical features")
    print(f"✓ Using {len(categorical_features)} categorical features")
    
    # ============================================================================
    # 3. PREPARE FEATURES AND TARGET
    # ============================================================================
    print("\n--- Step 2: Preparing Features and Target ---")
    
    all_feature_cols = numerical_features + categorical_features
    X = df[all_feature_cols].copy()
    y = df[TARGET_VARIABLE].copy()
    
    print(f"✓ Feature matrix X shape: {X.shape}")
    print(f"✓ Target vector y shape: {y.shape}")
    print(f"✓ Target distribution:\n{y.value_counts()}")
    print(f"✓ Target distribution (normalized):\n{y.value_counts(normalize=True)}")
    
    # ============================================================================
    # 4. SPLIT DATA
    # ============================================================================
    print("\n--- Step 3: Train-Test Split ---")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y  # Critical for imbalanced dataset
    )
    
    print(f"✓ Training set: {X_train.shape[0]} samples")
    print(f"✓ Test set: {X_test.shape[0]} samples")
    print(f"✓ Training target distribution:\n{y_train.value_counts()}")
    print(f"✓ Test target distribution:\n{y_test.value_counts()}")
    
    # ============================================================================
    # 5. FIX CLASS IMBALANCE
    # ============================================================================
    print("\n--- Step 4: Fixing Class Imbalance ---")
    
    # Calculate scale_pos_weight: ratio of majority class to minority class
    scale_pos_weight = y_train.value_counts()[0] / y_train.value_counts()[1]
    
    print(f"✓ Class Imbalance Ratio (scale_pos_weight): {scale_pos_weight:.2f}")
    print(f"  Majority class (0): {y_train.value_counts()[0]} samples")
    print(f"  Minority class (1): {y_train.value_counts()[1]} samples")
    
    # ============================================================================
    # 6. BUILD PREPROCESSING PIPELINE
    # ============================================================================
    print("\n--- Step 5: Building Preprocessing Pipeline ---")
    
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
    print("  - Numerical: median imputation + standardization")
    print("  - Categorical: mode imputation + one-hot encoding")
    
    # ============================================================================
    # 7. BUILD MODEL PIPELINE
    # ============================================================================
    print("\n--- Step 6: Building Model Pipeline ---")
    
    # Create the complete pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LGBMClassifier(
            random_state=42,
            scale_pos_weight=scale_pos_weight,  # Fix for class imbalance
            verbose=-1  # Suppress output
        ))
    ])
    
    print("✓ Model pipeline created with preprocessing and LGBMClassifier")
    
    # ============================================================================
    # 8. HYPERPARAMETER TUNING
    # ============================================================================
    print("\n--- Step 7: Hyperparameter Tuning ---")
    print("This may take several minutes...")
    
    # Define parameter distribution for hyperparameter tuning
    param_dist = {
        'classifier__n_estimators': [100, 200, 500],
        'classifier__learning_rate': [0.01, 0.05, 0.1],
        'classifier__num_leaves': [20, 31, 50],
        'classifier__max_depth': [5, 10, 15, -1],
        'classifier__min_child_samples': [20, 30, 50],
        'classifier__subsample': [0.8, 0.9, 1.0],
        'classifier__colsample_bytree': [0.8, 0.9, 1.0],
        'classifier__scale_pos_weight': [scale_pos_weight]  # Keep our calculated value
    }
    
    # Perform randomized search with cross-validation
    random_search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_dist,
        n_iter=25,
        cv=3,
        scoring='f1',  # Optimize for F1 score
        random_state=42,
        n_jobs=-1,
        verbose=1
    )
    
    try:
        random_search.fit(X_train, y_train)
        print("✓ Hyperparameter tuning completed successfully")
        print(f"✓ Best F1 score (CV): {random_search.best_score_:.4f}")
        print(f"✓ Best parameters: {random_search.best_params_}")
    except Exception as e:
        print(f"✗ ERROR: Hyperparameter tuning failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Get the best model
    tuned_model = random_search.best_estimator_
    
    # ============================================================================
    # 9. EVALUATE MODEL
    # ============================================================================
    print("\n" + "=" * 80)
    print("MODEL EVALUATION")
    print("=" * 80)
    
    # Get predictions on test set
    y_pred = tuned_model.predict(X_test)
    y_proba = tuned_model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)
    
    print(f"\n--- Performance Metrics (Test Set) ---")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"ROC AUC:   {roc_auc:.4f}")
    
    print(f"\n--- Classification Report ---")
    print(classification_report(y_test, y_pred, target_names=['No Dementia', 'Dementia']))
    
    print(f"\n--- Confusion Matrix ---")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    print(f"True Negatives:  {cm[0,0]}")
    print(f"False Positives: {cm[0,1]}")
    print(f"False Negatives: {cm[1,0]}")
    print(f"True Positives:  {cm[1,1]}")
    
    # ============================================================================
    # 10. FIND OPTIMAL THRESHOLD
    # ============================================================================
    print("\n" + "=" * 80)
    print("FINDING OPTIMAL DECISION THRESHOLD")
    print("=" * 80)
    
    # Get probability predictions for the training data
    y_train_proba = tuned_model.predict_proba(X_train)[:, 1]
    
    # Calculate precision, recall, and thresholds
    precision_curve, recall_curve, thresholds = precision_recall_curve(y_train, y_train_proba)
    
    # Find the F1 score for all thresholds
    # We add a small epsilon (1e-10) to avoid division by zero
    f1_scores = (2 * precision_curve * recall_curve) / (precision_curve + recall_curve + 1e-10)
    
    # Find the threshold that gives the highest F1 score
    best_f1_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_f1_idx]
    best_f1_score = f1_scores[best_f1_idx]
    
    print(f"✓ Best F1 Score on Train data: {best_f1_score:.4f}")
    print(f"✓ Optimal Decision Threshold: {best_threshold:.4f}")
    print(f"  (Default threshold would be 0.5)")
    
    # Evaluate with optimal threshold on test set
    y_pred_optimal = (y_proba >= best_threshold).astype(int)
    f1_optimal = f1_score(y_test, y_pred_optimal)
    
    print(f"\n--- Test Set Performance with Optimal Threshold ---")
    print(f"F1-Score: {f1_optimal:.4f}")
    print(f"  (vs {f1:.4f} with default 0.5 threshold)")
    
    # ============================================================================
    # 11. GENERATE SHAP PLOT (EXPLAINABILITY)
    # ============================================================================
    print("\n" + "=" * 80)
    print("GENERATING SHAP EXPLAINABILITY PLOT")
    print("=" * 80)
    
    if SHAP_AVAILABLE:
        try:
            print("Computing SHAP values (this may take a few minutes)...")
            
            # Get the preprocessed training data
            X_train_preprocessed = tuned_model.named_steps['preprocessor'].transform(X_train)
            
            # Get the classifier from the pipeline
            classifier = tuned_model.named_steps['classifier']
            
            # Create SHAP explainer
            # Use a sample of training data for faster computation
            sample_size = min(100, len(X_train_preprocessed))
            X_train_sample = X_train_preprocessed[:sample_size]
            
            print(f"Using {sample_size} samples for SHAP computation...")
            
            with suppress_stdout_stderr():
                explainer = shap.TreeExplainer(classifier)
                shap_values = explainer.shap_values(X_train_sample)
            
            # Create SHAP summary plot
            print("Generating SHAP summary plot...")
            plt.figure(figsize=(10, 8))
            
            # For binary classification, use the positive class (index 1)
            if isinstance(shap_values, list):
                shap_values_to_plot = shap_values[1]  # Use class 1
            else:
                shap_values_to_plot = shap_values
            
            shap.summary_plot(
                shap_values_to_plot,
                X_train_sample,
                feature_names=[f"Feature_{i}" for i in range(X_train_sample.shape[1])],
                show=False,
                max_display=20
            )
            
            plt.title("SHAP Summary Plot - Feature Importance", fontsize=14, fontweight='bold')
            plt.tight_layout()
            
            # Save the plot
            shap_plot_file = 'shap_summary_plot.png'
            plt.savefig(shap_plot_file, dpi=150, bbox_inches='tight')
            print(f"✓ SHAP plot saved to {shap_plot_file}")
            
            # Optionally display (comment out if running headless)
            # plt.show()
            plt.close()
            
        except Exception as e:
            print(f"⚠ WARNING: Failed to generate SHAP plot: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("⚠ SHAP library not available. Skipping explainability plot.")
    
    # ============================================================================
    # 12. SAVE ALL OUTPUTS
    # ============================================================================
    print("\n" + "=" * 80)
    print("SAVING MODEL AND CONFIGURATION")
    print("=" * 80)
    
    # Save the tuned model
    try:
        joblib.dump(tuned_model, MODEL_OUTPUT_FILE)
        print(f"✓ Model saved successfully to {MODEL_OUTPUT_FILE}")
    except Exception as e:
        print(f"✗ ERROR: Failed to save model: {e}")
        return
    
    # Save the optimal threshold and other config
    config = {
        'threshold': float(best_threshold),
        'scale_pos_weight': float(scale_pos_weight),
        'best_f1_score': float(best_f1_score),
        'test_f1_score': float(f1),
        'test_roc_auc': float(roc_auc),
        'test_f1_with_optimal_threshold': float(f1_optimal)
    }
    
    try:
        with open(CONFIG_OUTPUT_FILE, 'w') as f:
            json.dump(config, f, indent=4)
        print(f"✓ Configuration saved successfully to {CONFIG_OUTPUT_FILE}")
        print(f"  - Optimal threshold: {best_threshold:.4f}")
        print(f"  - Scale pos weight: {scale_pos_weight:.2f}")
        print(f"  - Test F1 score: {f1:.4f}")
    except Exception as e:
        print(f"✗ ERROR: Failed to save configuration: {e}")
        return
    
    # ============================================================================
    # FINAL SUMMARY
    # ============================================================================
    print("\n" + "=" * 80)
    print("TRAINING WORKFLOW COMPLETED SUCCESSFULLY")
    print("=" * 80)
    print(f"\nFinal Model Performance:")
    print(f"  Test F1 Score: {f1:.4f}")
    print(f"  Test ROC AUC:  {roc_auc:.4f}")
    print(f"  Optimal Threshold: {best_threshold:.4f}")
    print(f"\nOutput Files:")
    print(f"  - Model: {MODEL_OUTPUT_FILE}")
    print(f"  - Config: {CONFIG_OUTPUT_FILE}")
    if SHAP_AVAILABLE:
        print(f"  - SHAP Plot: shap_summary_plot.png")
    print("=" * 80)
    
    return tuned_model, X_test, y_test, best_threshold


if __name__ == "__main__":
    main()

