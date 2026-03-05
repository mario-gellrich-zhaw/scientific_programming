#!/usr/bin/env python3
"""
Week 02 Challenge - Titanic Dataset Solution
Complete solution to all 6 tasks
"""
import os
import sys
import pandas as pd
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

def main():
    print("="*70)
    print("WEEK 02 CHALLENGE - TITANIC DATASET SOLUTION")
    print("="*70)

    # ===== TASK 1: Verify kaggle.json =====
    print("\n[TASK 1] Verify kaggle.json in repository root")
    print("-"*70)

    repo_root = Path(__file__).parent.parent.parent
    kaggle_path = repo_root / 'kaggle.json'

    if kaggle_path.exists():
        print(f"✓ kaggle.json found at: {kaggle_path}")
    else:
        print(f"✗ kaggle.json NOT found at: {kaggle_path}")
        print(f"  Please add kaggle.json to the repository root")
        print(f"  Expected location: {kaggle_path}")
        return

    # ===== TASK 2: Download Titanic Dataset =====
    print("\n[TASK 2] Download Titanic Dataset using Kaggle API")
    print("-"*70)

    try:
        from kaggle.api.kaggle_api_extended import KaggleApi

        # Initialize and authenticate
        api = KaggleApi()
        api.authenticate()
        print("✓ Kaggle API authenticated successfully!")

        # Create data folder
        data_folder = Path(__file__).parent / 'data'
        data_folder.mkdir(exist_ok=True)
        print(f"✓ Data folder: {data_folder}")

        # Download dataset
        dataset_name = 'yasserh/titanic-dataset'
        api.dataset_download_files(dataset_name, path=str(data_folder), unzip=False)
        print(f"✓ Downloaded dataset: {dataset_name}")

        # List files
        print("\nFiles in data folder:")
        for file in sorted(data_folder.glob('*')):
            print(f"  - {file.name}")

    except ImportError:
        print("⚠ Kaggle API not available. Please install: pip3 install kaggle")
        print("  Using existing data if available...")
        data_folder = Path(__file__).parent / 'data'

    # ===== Load the Dataset =====
    print("\n[LOADING DATASET]")
    print("-"*70)

    data_folder = Path(__file__).parent / 'data'
    csv_files = list(data_folder.glob('*.csv'))

    if not csv_files:
        print("✗ No CSV files found in data folder!")
        return

    titanic_file = csv_files[0]
    print(f"Loading: {titanic_file.name}")

    df = pd.read_csv(titanic_file)
    print(f"✓ Dataset loaded: {df.shape[0]} rows × {df.shape[1]} columns")

    # ===== TASK 3: Identify Data Types =====
    print("\n[TASK 3] Identify Data Types in the Titanic Dataset")
    print("-"*70)
    print("\nData Types:")
    print(df.dtypes.to_string())
    print("\nData Type Summary:")
    print(df.dtypes.value_counts().to_string())

    # ===== TASK 4: Transform 'Sex' Variable to Binary Columns =====
    print("\n[TASK 4] Transform 'Sex' Variable to Binary Columns")
    print("-"*70)

    print("\nOriginal 'Sex' column unique values:")
    print(df['Sex'].unique())
    print(f"\nValue counts:\n{df['Sex'].value_counts()}")

    # Method 1: Using get_dummies
    print("\n--- Method 1: Using get_dummies ---")
    sex_dummies = pd.get_dummies(df['Sex'], prefix='Sex', drop_first=False)
    print("Binary matrix (first 10 rows):")
    print(sex_dummies.head(10))
    print(f"\nShape: {sex_dummies.shape}")

    # Method 2: Manual binary encoding
    print("\n--- Method 2: Manual Binary Encoding ---")
    df['Sex_Female'] = (df['Sex'] == 'female').astype(int)
    df['Sex_Male'] = (df['Sex'] == 'male').astype(int)

    print("Binary columns created:")
    print(df[['Sex', 'Sex_Female', 'Sex_Male']].head(10).to_string())

    print(f"\nSex_Female value counts:\n{df['Sex_Female'].value_counts()}")
    print(f"\nSex_Male value counts:\n{df['Sex_Male'].value_counts()}")

    # ===== TASK 5 & 6: Create Subset and Count Passengers =====
    print("\n[TASK 5 & 6] Create Subset with Specific Conditions")
    print("-"*70)

    print("\nConditions:")
    print("  - Survived = 1 (passengers who survived)")
    print("  AND")
    print("  - (Female AND Age > 45) OR (Male AND Age < 20)")

    # Create the subset
    condition = (
        (df['Survived'] == 1) &
        (
            ((df['Sex'] == 'female') & (df['Age'] > 45)) |
            ((df['Sex'] == 'male') & (df['Age'] < 20))
        )
    )

    subset = df[condition].copy()

    print(f"\n✓ Subset created successfully")
    print(f"  Total rows in subset: {len(subset)}")

    # Detailed breakdown
    print("\n" + "="*70)
    print("DETAILED ANALYSIS OF THE SUBSET")
    print("="*70)

    print(f"\nTotal passengers in subset: {len(subset)}")

    print(f"\nGender distribution:")
    print(subset['Sex'].value_counts().to_string())

    females_45plus = subset[subset['Sex'] == 'female']
    print(f"\nFemales (Age > 45) in subset:")
    print(f"  Count: {len(females_45plus)}")
    if len(females_45plus) > 0:
        print(f"  Age range: {females_45plus['Age'].min():.0f} to {females_45plus['Age'].max():.0f} years")
        print(f"  Mean age: {females_45plus['Age'].mean():.2f} years")

    males_under20 = subset[subset['Sex'] == 'male']
    print(f"\nMales (Age < 20) in subset:")
    print(f"  Count: {len(males_under20)}")
    if len(males_under20) > 0:
        print(f"  Age range: {males_under20['Age'].min():.0f} to {males_under20['Age'].max():.0f} years")
        print(f"  Mean age: {males_under20['Age'].mean():.2f} years")

    # Final Answer
    print("\n" + "#"*70)
    print("# ANSWER TO QUESTION 6: How many passengers were selected?")
    print("#"*70)
    print(f"\n>>> {len(subset)} passengers were selected <<<\n")
    print("#"*70)

    # Display all selected passengers
    print("\nFull list of selected passengers:")
    print("-"*70)

    output_columns = ['PassengerId', 'Name', 'Sex', 'Age', 'Survived', 'Pclass']
    available_columns = [col for col in output_columns if col in subset.columns]

    print(subset[available_columns].to_string(index=False))

    # Save results to file
    output_file = Path(__file__).parent / 'challenge_results.txt'
    with open(output_file, 'w') as f:
        f.write("WEEK 02 CHALLENGE - TITANIC DATASET RESULTS\n")
        f.write("="*70 + "\n\n")
        f.write(f"ANSWER TO QUESTION 6: {len(subset)} passengers were selected\n\n")
        f.write("Selected Passengers:\n")
        f.write("-"*70 + "\n")
        f.write(subset[available_columns].to_string(index=False))

    print(f"\n✓ Results saved to: {output_file}")

    print("\n" + "="*70)
    print("CHALLENGE COMPLETE!")
    print("="*70)

if __name__ == '__main__':
    main()

