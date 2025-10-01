#!/usr/bin/env python3
"""
Example script demonstrating student grade prediction using Random Forest.

This script shows a simplified version of the analysis performed in Project.ipynb.
It loads the student data, performs basic preprocessing, trains a Random Forest model,
and evaluates its performance.

Usage:
    python example.py
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings

warnings.filterwarnings('ignore')


def load_and_preprocess_data(filepath='data/student-mat.csv'):
    """Load and preprocess the student performance dataset."""
    print("Loading data...")
    data = pd.read_csv(filepath, sep=';')

    print(f"Dataset shape: {data.shape}")
    print(f"Columns: {list(data.columns)}\n")

    # Encode categorical variables
    print("Encoding categorical variables...")
    categorical_cols = data.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        data[col], _ = pd.factorize(data[col])

    # Create performance failure feature
    print("Creating engineered features...")
    data['performanceFailure'] = (data['G1'] + data['G2']) / 2 + data['failures']

    # Normalize age and absences
    scaler = MinMaxScaler()
    data[['age', 'absences']] = scaler.fit_transform(data[['age', 'absences']])

    # Drop G1 and G2 (highly correlated with target)
    features_to_drop = ['G1', 'G2']
    X = data.drop(columns=features_to_drop + ['G3'])
    y = data['G3']

    print(f"Final feature set: {X.shape[1]} features")
    print(f"Target variable (G3) range: [{y.min()}, {y.max()}]\n")

    return X, y


def train_and_evaluate(X, y):
    """Train Random Forest model and evaluate performance."""
    print("Splitting data into train and test sets (80-20 split)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"Training set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}\n")

    print("Training Random Forest Regressor...")
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    print("Making predictions...")
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Calculate metrics
    print("\n" + "="*50)
    print("MODEL PERFORMANCE")
    print("="*50)

    print("\nTraining Set:")
    print(f"  R² Score: {r2_score(y_train, y_train_pred):.4f}")
    print(f"  MAE: {mean_absolute_error(y_train, y_train_pred):.4f}")
    print(f"  RMSE: {np.sqrt(mean_squared_error(y_train, y_train_pred)):.4f}")

    print("\nTest Set:")
    print(f"  R² Score: {r2_score(y_test, y_test_pred):.4f}")
    print(f"  MAE: {mean_absolute_error(y_test, y_test_pred):.4f}")
    print(f"  RMSE: {np.sqrt(mean_squared_error(y_test, y_test_pred)):.4f}")

    # Feature importance
    print("\n" + "="*50)
    print("TOP 10 MOST IMPORTANT FEATURES")
    print("="*50)
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    print(feature_importance.head(10).to_string(index=False))
    print("\n")

    return model, (y_test, y_test_pred)


def main():
    """Main execution function."""
    print("="*50)
    print("STUDENT GRADE PREDICTION - EXAMPLE SCRIPT")
    print("="*50)
    print()

    # Load and preprocess data
    X, y = load_and_preprocess_data()

    # Train and evaluate model
    model, (y_test, y_test_pred) = train_and_evaluate(X, y)

    print("="*50)
    print("EXAMPLE COMPLETED SUCCESSFULLY!")
    print("="*50)
    print("\nFor a more detailed analysis, please refer to:")
    print("  - code/Project.ipynb (Jupyter Notebook)")
    print("  - docs/Report_project_Leo_Zixuan.pdf (Full Report)")
    print()


if __name__ == "__main__":
    main()
