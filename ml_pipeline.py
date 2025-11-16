"""
Complete Machine Learning Pipeline for Dementia Prediction
Author: Senior Data Scientist
Purpose: Build a full ML pipeline to handle missing data and train a binary classification model
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


def identify_feature_types(df, target_col, numerical_threshold=20):
    """
    Automatically identify numerical vs categorical features.
    
    Parameters:
    -----------
    df : DataFrame
        The input dataframe
    target_col : str
        Name of the target column to exclude
    numerical_threshold : int
        Maximum number of unique values for a feature to be considered categorical
    
    Returns:
    --------
    numerical_features : list
        List of numerical feature names
    categorical_features : list
        List of categorical feature names
    """
    numerical_features = []
    categorical_features = []
    
    # Exclude target variable
    feature_cols = [col for col in df.columns if col != target_col]
    
    for col in feature_cols:
        # Skip if all values are NaN
        if df[col].isna().all():
            continue
            
        # Check data type
        if df[col].dtype in ['int64', 'float64']:
            # Count non-null unique values
            n_unique = df[col].nunique()
            
            # If it has many unique values, it's likely numerical
            # If it has few unique values, it might be categorical (encoded)
            if n_unique > numerical_threshold:
                numerical_features.append(col)
            else:
                # Additional check: if values are mostly integers in a small range, might be categorical
                non_null_values = df[col].dropna()
                if len(non_null_values) > 0:
                    if non_null_values.dtype == 'float64' and (non_null_values % 1 == 0).all():
                        # All values are whole numbers
                        if n_unique <= numerical_threshold:
                            categorical_features.append(col)
                        else:
                            numerical_features.append(col)
                    else:
                        numerical_features.append(col)
                else:
                    numerical_features.append(col)
        else:
            # Object/string type is categorical
            categorical_features.append(col)
    
    return numerical_features, categorical_features


def main():
    """
    Main function to execute the complete ML pipeline.
    """
    
    # ============================================================================
    # STEP 1: Load Data
    # ============================================================================
    print("=" * 80)
    print("MACHINE LEARNING PIPELINE FOR DEMENTIA PREDICTION")
    print("=" * 80)
    print("\n--- Step 1: Loading Data ---")
    
    INPUT_FILE = 'cleaned_Dementia_Prediction_Dataset.csv'
    
    try:
        df = pd.read_csv(INPUT_FILE)
        print(f"✓ Successfully loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    except FileNotFoundError:
        print(f"✗ ERROR: File '{INPUT_FILE}' not found.")
        return
    except Exception as e:
        print(f"✗ ERROR: Failed to load data: {e}")
        return
    
    # ============================================================================
    # STEP 2: Target and Feature Definitions
    # ============================================================================
    print("\n--- Step 2: Defining Target and Features ---")
    
    TARGET = 'DEMENTED'
    
    # Verify target exists
    if TARGET not in df.columns:
        print(f"✗ ERROR: Target variable '{TARGET}' not found in dataset.")
        return
    
    # Automatically identify numerical and categorical features
    numerical_features, categorical_features = identify_feature_types(df, TARGET, numerical_threshold=20)
    
    # Self-correction: Ensure DEMENTED is not in feature lists
    if TARGET in numerical_features:
        numerical_features.remove(TARGET)
    if TARGET in categorical_features:
        categorical_features.remove(TARGET)
    
    print(f"✓ Identified {len(numerical_features)} numerical features")
    print(f"✓ Identified {len(categorical_features)} categorical features")
    print(f"\nNumerical features: {numerical_features}")
    print(f"\nCategorical features: {categorical_features}")
    
    # ============================================================================
    # STEP 3: Define X and y
    # ============================================================================
    print("\n--- Step 3: Preparing Features and Target ---")
    
    # Combine all feature columns
    all_feature_cols = numerical_features + categorical_features
    
    # Ensure all columns exist
    missing_cols = [col for col in all_feature_cols if col not in df.columns]
    if missing_cols:
        print(f"⚠ WARNING: Some features are missing: {missing_cols}")
        all_feature_cols = [col for col in all_feature_cols if col in df.columns]
        numerical_features = [col for col in numerical_features if col in df.columns]
        categorical_features = [col for col in categorical_features if col in df.columns]
    
    X = df[all_feature_cols].copy()
    y = df[TARGET].copy()
    
    print(f"✓ Feature matrix X shape: {X.shape}")
    print(f"✓ Target vector y shape: {y.shape}")
    print(f"✓ Target distribution:\n{y.value_counts(normalize=True)}")
    
    # ============================================================================
    # STEP 4: Train-Test Split
    # ============================================================================
    print("\n--- Step 4: Train-Test Split ---")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y  # Critical for imbalanced dataset
    )
    
    print(f"✓ Training set: {X_train.shape[0]} samples")
    print(f"✓ Test set: {X_test.shape[0]} samples")
    print(f"✓ Training target distribution:\n{y_train.value_counts(normalize=True)}")
    print(f"✓ Test target distribution:\n{y_test.value_counts(normalize=True)}")
    
    # Calculate scale_pos_weight for class imbalance
    scale_pos_weight = y_train.value_counts()[0] / y_train.value_counts()[1]
    print(f"\n✓ Class Imbalance Ratio (scale_pos_weight): {scale_pos_weight:.2f}")
    
    # ============================================================================
    # STEP 5: Build Preprocessing Pipelines
    # ============================================================================
    print("\n--- Step 5: Building Preprocessing Pipelines ---")
    
    # Numerical transformer pipeline
    # Step 1: Fill empty cells with median
    # Step 2: Scale all numbers
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Categorical transformer pipeline
    # Step 1: Fill empty cells with most frequent value (mode)
    # Step 2: Convert categories to numbers using one-hot encoding
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    print("✓ Created numerical transformer (median imputation + standardization)")
    print("✓ Created categorical transformer (mode imputation + one-hot encoding)")
    
    # ============================================================================
    # STEP 6: Create ColumnTransformer
    # ============================================================================
    print("\n--- Step 6: Creating Column Transformer ---")
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='drop'  # Drop any columns not explicitly handled
    )
    
    print("✓ ColumnTransformer created")
    
    # ============================================================================
    # STEP 7: Create Full Model Pipeline
    # ============================================================================
    print("\n--- Step 7: Creating Full Model Pipeline ---")
    
    # Create the complete pipeline
    # Step 1: Preprocessing (ColumnTransformer)
    # Step 2: Model (LGBMClassifier)
    full_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LGBMClassifier(
            random_state=42,
            scale_pos_weight=scale_pos_weight,  # Handle imbalanced data
            verbose=-1  # Suppress output
        ))
    ])
    
    print("✓ Full pipeline created with preprocessing and LGBMClassifier")
    
    # ============================================================================
    # STEP 8: Hyperparameter Tuning
    # ============================================================================
    print("\n--- Step 8: Hyperparameter Tuning ---")
    print("This may take several minutes...")
    
    # Define parameter distribution for hyperparameter tuning
    param_dist = {
        'classifier__n_estimators': [100, 200, 300],
        'classifier__max_depth': [5, 10, 15, -1],
        'classifier__learning_rate': [0.01, 0.05, 0.1, 0.2],
        'classifier__num_leaves': [31, 50, 100],
        'classifier__min_child_samples': [20, 30, 50],
        'classifier__subsample': [0.8, 0.9, 1.0],
        'classifier__colsample_bytree': [0.8, 0.9, 1.0],
        'classifier__scale_pos_weight': [scale_pos_weight]  # Use calculated scale_pos_weight
    }
    
    # Perform randomized search with cross-validation
    random_search = RandomizedSearchCV(
        estimator=full_pipeline,
        param_distributions=param_dist,
        n_iter=20,  # Number of parameter settings sampled
        cv=3,  # 3-fold cross-validation
        scoring='f1',  # Use F1 score for imbalanced data
        n_jobs=-1,  # Use all available cores
        random_state=42,
        verbose=1
    )
    
    try:
        random_search.fit(X_train, y_train)
        print("✓ Hyperparameter tuning completed successfully")
        print(f"✓ Best F1 score: {random_search.best_score_:.4f}")
        print(f"✓ Best parameters: {random_search.best_params_}")
    except Exception as e:
        print(f"✗ ERROR: Hyperparameter tuning failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Get the best model
    tuned_model = random_search.best_estimator_
    
    # ============================================================================
    # STEP 9: Evaluate the Model
    # ============================================================================
    print("\n" + "=" * 80)
    print("MODEL EVALUATION REPORT")
    print("=" * 80)
    
    # Get predictions using the tuned model
    y_pred = tuned_model.predict(X_test)
    
    # Get prediction probabilities
    y_proba = tuned_model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)
    
    # Print evaluation metrics
    print(f"\n--- Performance Metrics ---")
    print(f"F1-Score: {f1:.4f}")
    print(f"ROC AUC Score: {roc_auc:.4f}")
    
    # Additional useful metrics
    from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report, confusion_matrix
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    
    print(f"\n--- Additional Metrics ---")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    
    print(f"\n--- Classification Report ---")
    print(classification_report(y_test, y_pred, target_names=['No Dementia', 'Dementia']))
    
    print(f"\n--- Confusion Matrix ---")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    print(f"True Negatives: {cm[0,0]}, False Positives: {cm[0,1]}")
    print(f"False Negatives: {cm[1,0]}, True Positives: {cm[1,1]}")
    
    # ============================================================================
    # STEP 11: Save the Final Tuned Model
    # ============================================================================
    print("\n" + "=" * 80)
    print("SAVING FINAL TUNED MODEL")
    print("=" * 80)
    
    MODEL_FILE = 'dementia_model.joblib'
    try:
        joblib.dump(tuned_model, MODEL_FILE)
        print(f"✓ Model saved successfully to {MODEL_FILE}")
        print("\n--- Final Tuned Model Saved to dementia_model.joblib ---")
    except Exception as e:
        print(f"✗ ERROR: Failed to save model: {e}")
    
    print("\n" + "=" * 80)
    print("PIPELINE EXECUTION COMPLETED SUCCESSFULLY")
    print("=" * 80)
    
    return tuned_model, X_test, y_test


if __name__ == "__main__":
    main()

