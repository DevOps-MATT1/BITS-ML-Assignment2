"""
Test Data Generator for Loan Approval Prediction
ML Assignment 2 - Sumit Mondal (2024dc04216)

This script generates realistic synthetic test data for loan approval prediction.
"""

import pandas as pd
import numpy as np
from datetime import datetime

def generate_test_data(num_samples=100, output_file='test_data.csv', include_labels=True):
    """
    Generate synthetic test data for loan approval prediction
    
    Parameters:
    -----------
    num_samples : int
        Number of test samples to generate (default: 100)
    output_file : str
        Output CSV filename (default: 'test_data.csv')
    include_labels : bool
        Whether to include loan_status labels (default: True)
    
    Returns:
    --------
    pd.DataFrame
        Generated test data
    """
    
    np.random.seed(42)  # For reproducibility
    
    # Define categorical options
    genders = ['male', 'female']
    educations = ['High School', 'Bachelor', 'Master', 'Associate', 'Doctorate']
    home_ownerships = ['RENT', 'OWN', 'MORTGAGE', 'OTHER']
    loan_intents = ['PERSONAL', 'EDUCATION', 'MEDICAL', 'VENTURE', 
                    'HOMEIMPROVEMENT', 'DEBTCONSOLIDATION']
    defaults = ['No', 'Yes']
    
    # Generate data
    data = {
        # Personal Information
        'person_age': np.random.randint(20, 70, num_samples).astype(float),
        'person_gender': np.random.choice(genders, num_samples),
        'person_education': np.random.choice(educations, num_samples, 
                                            p=[0.25, 0.35, 0.25, 0.10, 0.05]),
        
        # Financial Information
        'person_income': np.random.lognormal(10.8, 0.6, num_samples).astype(float),
        'person_emp_exp': np.random.randint(0, 40, num_samples),
        'person_home_ownership': np.random.choice(home_ownerships, num_samples,
                                                 p=[0.40, 0.30, 0.25, 0.05]),
        
        # Loan Details
        'loan_amnt': np.random.choice(
            [500, 1000, 2500, 5000, 7500, 10000, 15000, 20000, 25000, 30000, 35000],
            num_samples
        ).astype(float),
        'loan_intent': np.random.choice(loan_intents, num_samples),
        'loan_int_rate': np.random.uniform(5.42, 20.0, num_samples).round(2),
        
        # Credit Information
        'cb_person_cred_hist_length': np.random.uniform(2.0, 30.0, num_samples).round(1),
        'credit_score': np.random.randint(390, 850, num_samples),
        'previous_loan_defaults_on_file': np.random.choice(defaults, num_samples,
                                                          p=[0.80, 0.20])
    }
    
    # Calculate loan_percent_income
    data['loan_percent_income'] = (data['loan_amnt'] / data['person_income']).round(2)
    
    # Clip loan_percent_income to realistic range
    data['loan_percent_income'] = np.clip(data['loan_percent_income'], 0.0, 0.66)
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Reorder columns to match expected format
    column_order = [
        'person_age', 'person_gender', 'person_education', 'person_income',
        'person_emp_exp', 'person_home_ownership', 'loan_amnt', 'loan_intent',
        'loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length',
        'credit_score', 'previous_loan_defaults_on_file'
    ]
    
    df = df[column_order]
    
    # Generate realistic labels if requested
    if include_labels:
        # Simple heuristic for generating labels (not perfect, but realistic)
        # Higher chance of approval with:
        # - Higher credit score
        # - Lower loan-to-income ratio
        # - No previous defaults
        # - Longer credit history
        
        approval_score = (
            (df['credit_score'] - 390) / (850 - 390) * 0.4 +  # 40% weight
            (1 - df['loan_percent_income'] / 0.66) * 0.25 +    # 25% weight
            (df['previous_loan_defaults_on_file'] == 'No').astype(int) * 0.20 +  # 20% weight
            (df['cb_person_cred_hist_length'] / 30.0) * 0.15   # 15% weight
        )
        
        # Add some randomness
        approval_score += np.random.normal(0, 0.1, num_samples)
        
        # Convert to binary labels (threshold at 0.5)
        df['loan_status'] = (approval_score > 0.5).astype(int)
    
    # Save to CSV
    df.to_csv(output_file, index=False)
    print(f"✅ Generated {num_samples} test samples")
    print(f"📁 Saved to: {output_file}")
    
    if include_labels:
        approval_rate = df['loan_status'].mean() * 100
        print(f"📊 Approval Rate: {approval_rate:.1f}%")
        print(f"   - Approved: {df['loan_status'].sum()}")
        print(f"   - Rejected: {len(df) - df['loan_status'].sum()}")
    
    return df


def generate_multiple_datasets():
    """Generate multiple test datasets for different scenarios"""
    
    print("=" * 70)
    print("GENERATING TEST DATASETS FOR LOAN APPROVAL PREDICTION")
    print("=" * 70)
    print()
    
    # 1. Small test set with labels (for evaluation)
    print("1️⃣  Generating small test set (50 samples) with labels...")
    df_small = generate_test_data(
        num_samples=50,
        output_file='test_data_small.csv',
        include_labels=True
    )
    print()
    
    # 2. Medium test set with labels (for evaluation)
    print("2️⃣  Generating medium test set (200 samples) with labels...")
    df_medium = generate_test_data(
        num_samples=200,
        output_file='test_data_medium.csv',
        include_labels=True
    )
    print()
    
    # 3. Large test set with labels (for evaluation)
    print("3️⃣  Generating large test set (500 samples) with labels...")
    df_large = generate_test_data(
        num_samples=500,
        output_file='test_data_large.csv',
        include_labels=True
    )
    print()
    
    # 4. Prediction-only dataset (without labels)
    print("4️⃣  Generating prediction-only dataset (100 samples)...")
    df_pred = generate_test_data(
        num_samples=100,
        output_file='prediction_data.csv',
        include_labels=False
    )
    print()
    
    # 5. Edge cases dataset
    print("5️⃣  Generating edge cases dataset...")
    generate_edge_cases()
    print()
    
    print("=" * 70)
    print("✅ ALL TEST DATASETS GENERATED SUCCESSFULLY!")
    print("=" * 70)
    print()
    print("📁 Generated files:")
    print("   - test_data_small.csv (50 samples with labels)")
    print("   - test_data_medium.csv (200 samples with labels)")
    print("   - test_data_large.csv (500 samples with labels)")
    print("   - prediction_data.csv (100 samples without labels)")
    print("   - edge_cases.csv (special test cases)")


def generate_edge_cases():
    """Generate edge cases for testing model robustness"""
    
    edge_cases = {
        'person_age': [20, 20, 25, 30, 40, 50, 60, 69, 25, 35],
        'person_gender': ['male', 'female', 'male', 'female', 'male', 
                         'female', 'male', 'female', 'male', 'female'],
        'person_education': ['High School', 'Doctorate', 'Bachelor', 'Master',
                            'Associate', 'High School', 'Bachelor', 'Master',
                            'Doctorate', 'Bachelor'],
        'person_income': [8000, 500000, 25000, 75000, 50000, 
                         100000, 40000, 200000, 15000, 80000],
        'person_emp_exp': [0, 25, 2, 10, 15, 20, 5, 30, 1, 12],
        'person_home_ownership': ['RENT', 'OWN', 'MORTGAGE', 'RENT', 'OWN',
                                 'MORTGAGE', 'OTHER', 'OWN', 'RENT', 'MORTGAGE'],
        'loan_amnt': [500, 35000, 5000, 20000, 10000, 
                     25000, 15000, 30000, 1000, 12000],
        'loan_intent': ['PERSONAL', 'VENTURE', 'EDUCATION', 'MEDICAL',
                       'HOMEIMPROVEMENT', 'DEBTCONSOLIDATION', 'PERSONAL',
                       'EDUCATION', 'MEDICAL', 'VENTURE'],
        'loan_int_rate': [5.42, 19.99, 8.5, 12.0, 10.5,
                         15.0, 7.5, 18.0, 6.0, 11.0],
        'loan_percent_income': [0.06, 0.07, 0.20, 0.27, 0.20,
                               0.25, 0.38, 0.15, 0.07, 0.15],
        'cb_person_cred_hist_length': [2.0, 30.0, 5.0, 10.0, 8.0,
                                       15.0, 3.0, 25.0, 2.5, 12.0],
        'credit_score': [390, 850, 600, 720, 650,
                        700, 580, 780, 500, 680],
        'previous_loan_defaults_on_file': ['Yes', 'No', 'No', 'No', 'Yes',
                                          'No', 'Yes', 'No', 'Yes', 'No'],
        'loan_status': [0, 1, 1, 1, 0, 1, 0, 1, 0, 1]
    }
    
    df = pd.DataFrame(edge_cases)
    df.to_csv('edge_cases.csv', index=False)
    print(f"✅ Generated {len(df)} edge case samples")
    print(f"📁 Saved to: edge_cases.csv")


if __name__ == "__main__":
    # Generate all test datasets
    generate_multiple_datasets()
    
    # Display sample from one of the generated files
    print("\n📋 Sample from test_data_medium.csv:")
    print("=" * 70)
    df_sample = pd.read_csv('test_data_medium.csv')
    print(df_sample.head(10).to_string(index=False))
    print()
    print(f"Total samples: {len(df_sample)}")
    print(f"Columns: {', '.join(df_sample.columns)}")
