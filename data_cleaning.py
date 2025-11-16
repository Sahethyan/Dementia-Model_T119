"""
Comprehensive Data Loading and Cleaning Script for Dementia Prediction Dataset
Author: Senior ML Engineer
Purpose: Load, filter, and clean the Dementia_Prediction_Dataset.csv file
"""

import pandas as pd
import numpy as np


def main():
    """
    Main function to perform data loading and cleaning operations.
    """
    
    # ============================================================================
    # STEP 1: Define Constants
    # ============================================================================
    INPUT_FILE = 'Dementia_Prediction_Dataset.csv'
    OUTPUT_FILE = 'cleaned_Dementia_Prediction_Dataset.csv'
    TARGET_VARIABLE = 'DEMENTED'
    
    # Define allowed features from different forms
    # Form A1 features
    form_a1_features = [
        'NACCREAS', 'NACCREFR', 'BIRTHMO', 'BIRTHYR', 'SEX', 'HISPANIC', 
        'HISPOR', 'RACE', 'RACESEC', 'RACETER', 'PRIMLANG', 'EDUC', 
        'MARISTAT', 'NACCLIVS', 'INDEPEND', 'RESIDENC', 'HANDED', 
        'NACCAGE', 'NACCAGEB', 'NACCNIHR'
    ]
    
    # Form A2 features
    form_a2_features = [
        'INBIRMO', 'INBIRYR', 'INSEX', 'NEWINF', 'INHISP', 'INHISPOR', 
        'INRACE', 'INRASEC', 'INRATER', 'INEDUC', 'INRELTO', 'INKNOWN', 
        'INLIVWTH', 'INVISITS', 'INCALLS', 'INRELY', 'NACCNINR'
    ]
    
    # Form CLS features
    form_cls_features = [
        'APREFLAN', 'AYRSPAN', 'AYRENGL', 'APCSPAN', 'APCENGL', 'ASPKSPAN', 
        'AREASPAN', 'AWRISPAN', 'AUNDSPAN', 'ASPKENGL', 'AREAENGL', 
        'AWRIENGL', 'AUNDENGL', 'NACCSPNL', 'NACCENGL'
    ]
    
    # Form Header & Milestones features
    form_header_milestones_features = [
        'VISITMO', 'VISITDAY', 'VISITYR', 'NACCVNUM', 'NACCAVST', 
        'NACCNVST', 'NACCDAYS', 'NACCFDYS'
    ]
    
    # Combine all allowed features into a single list
    ALLOWED_FEATURES = (
        form_a1_features + 
        form_a2_features + 
        form_cls_features + 
        form_header_milestones_features
    )
    
    # ============================================================================
    # STEP 2: Load and Filter Data
    # ============================================================================
    print("=" * 80)
    print("DATA LOADING AND CLEANING PROCESS")
    print("=" * 80)
    print(f"\nLoading data from: {INPUT_FILE}")
    print(f"Target variable: {TARGET_VARIABLE}")
    print(f"Number of allowed features (requested): {len(ALLOWED_FEATURES)}")
    
    try:
        # First, read just the header to check which columns exist
        available_columns = pd.read_csv(INPUT_FILE, nrows=0).columns.tolist()
        available_columns_set = set(available_columns)
        
        # Filter allowed features to only those that exist in the dataset
        ALLOWED_FEATURES_EXISTING = [f for f in ALLOWED_FEATURES if f in available_columns_set]
        MISSING_FEATURES = [f for f in ALLOWED_FEATURES if f not in available_columns_set]
        
        if MISSING_FEATURES:
            print(f"\n⚠ WARNING: {len(MISSING_FEATURES)} requested features are not in the dataset:")
            for feat in MISSING_FEATURES:
                print(f"  - {feat}")
        
        print(f"Number of allowed features (available): {len(ALLOWED_FEATURES_EXISTING)}")
        
        # Check if target variable exists
        if TARGET_VARIABLE not in available_columns_set:
            print(f"\n✗ ERROR: Target variable '{TARGET_VARIABLE}' not found in dataset!")
            return
        
        # Create final column list including target variable (only existing columns)
        COLUMNS_TO_KEEP = ALLOWED_FEATURES_EXISTING + [TARGET_VARIABLE]
        print(f"Total columns to keep: {len(COLUMNS_TO_KEEP)}")
        
        # Load only the required columns for memory efficiency
        df = pd.read_csv(INPUT_FILE, usecols=COLUMNS_TO_KEEP)
        print(f"\n✓ Successfully loaded data with {len(df)} rows and {len(df.columns)} columns")
    except FileNotFoundError:
        print(f"\n✗ ERROR: File '{INPUT_FILE}' not found in the current directory.")
        print("Please ensure the file exists in the same directory as this script.")
        return
    except Exception as e:
        print(f"\n✗ ERROR: An unexpected error occurred while loading the file: {e}")
        return
    
    # ============================================================================
    # STEP 3: Clean Special Codes
    # ============================================================================
    print("\n" + "-" * 80)
    print("CLEANING SPECIAL CODES")
    print("-" * 80)
    
    # Define all known special codes that represent missing/unknown/not-applicable data
    codes_to_replace = [-4, 8, 88, 888, 8888, 9, 99, 999, 9999]
    
    print(f"Replacing special codes {codes_to_replace} with NaN...")
    
    # Count missing values before cleaning
    missing_before = df.isnull().sum().sum()
    
    # Replace special codes with NaN
    df.replace(codes_to_replace, np.nan, inplace=True)
    
    # Count missing values after cleaning
    missing_after = df.isnull().sum().sum()
    codes_replaced = missing_after - missing_before
    
    print(f"✓ Replaced {codes_replaced} special code values with NaN")
    
    # ============================================================================
    # STEP 4: Post-Cleaning Analysis
    # ============================================================================
    print("\n" + "=" * 80)
    print("DATA CLEANING REPORT")
    print("=" * 80)
    
    # Print DataFrame info
    print("\n--- DataFrame Information ---")
    df.info()
    
    # Print missing values count for all columns
    print("\n--- Missing Values Count by Column ---")
    missing_counts = df.isnull().sum()
    print(missing_counts)
    
    # Print summary statistics for missing values
    print(f"\n--- Missing Values Summary ---")
    print(f"Total missing values: {missing_counts.sum()}")
    print(f"Columns with missing values: {(missing_counts > 0).sum()}")
    print(f"Columns without missing values: {(missing_counts == 0).sum()}")
    
    # Print class distribution of target variable
    print(f"\n--- Target Variable Distribution ({TARGET_VARIABLE}) ---")
    if TARGET_VARIABLE in df.columns:
        target_dist = df[TARGET_VARIABLE].value_counts(normalize=True)
        target_counts = df[TARGET_VARIABLE].value_counts()
        print("Normalized distribution (proportions):")
        print(target_dist)
        print("\nAbsolute counts:")
        print(target_counts)
        
        # Check for missing values in target
        target_missing = df[TARGET_VARIABLE].isnull().sum()
        if target_missing > 0:
            print(f"\n⚠ WARNING: {target_missing} missing values found in target variable!")
    else:
        print(f"⚠ WARNING: Target variable '{TARGET_VARIABLE}' not found in dataset!")
    
    # Print final dataset shape
    print(f"\n--- Final Dataset Shape ---")
    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")
    
    # ============================================================================
    # STEP 5: Save Cleaned Data
    # ============================================================================
    print("\n" + "-" * 80)
    print("SAVING CLEANED DATA")
    print("-" * 80)
    
    try:
        df.to_csv(OUTPUT_FILE, index=False)
        print(f"✓ Successfully saved cleaned data to: {OUTPUT_FILE}")
        print(f"  - Rows: {len(df)}")
        print(f"  - Columns: {len(df.columns)}")
    except Exception as e:
        print(f"✗ ERROR: Failed to save cleaned data: {e}")
        return
    
    print("\n" + "=" * 80)
    print("DATA CLEANING PROCESS COMPLETED SUCCESSFULLY")
    print("=" * 80)


if __name__ == "__main__":
    main()

