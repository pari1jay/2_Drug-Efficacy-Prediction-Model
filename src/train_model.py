"""
Drug Efficacy Prediction Model Training
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import pickle
import os

def create_sample_drug_data():
    """Create realistic sample drug data for training"""
    np.random.seed(42)
    n_samples = 1000
    
    # Generate synthetic drug features
    data = {
        'molecular_weight': np.random.normal(300, 100, n_samples),
        'logp': np.random.normal(2.5, 1.5, n_samples),  # Lipophilicity
        'polar_surface_area': np.random.normal(80, 30, n_samples),
        'num_rotatable_bonds': np.random.poisson(5, n_samples),
        'num_aromatic_rings': np.random.poisson(2, n_samples),
        'num_hydrogen_donors': np.random.poisson(2, n_samples),
        'num_hydrogen_acceptors': np.random.poisson(4, n_samples),
    }
    
    df = pd.DataFrame(data)
    
    # Create realistic efficacy score based on features
    # This simulates how molecular properties affect drug efficacy
    efficacy = (
        0.3 * (df['logp'] - 2.5) / 2.0 +  # Optimal LogP around 2.5
        -0.2 * (df['molecular_weight'] - 300) / 200 +  # Prefer moderate MW
        -0.1 * df['polar_surface_area'] / 100 +
        0.2 * df['num_aromatic_rings'] / 3 +
        np.random.normal(0, 0.1, n_samples)  # Add noise
    )
    
    # Convert to 0-1 scale and clip
    df['efficacy_score'] = np.clip((efficacy + 1) / 2, 0, 1)
    
    return df

def train_drug_efficacy_model():
    """Train a machine learning model to predict drug efficacy"""
    print("=== Drug Efficacy Prediction Model Training ===")
    
    # Create sample data
    print("1. Generating synthetic drug data...")
    df = create_sample_drug_data()
    print(f"   Generated {len(df)} drug samples")
    print(f"   Features: {list(df.columns[:-1])}")
    
    # Prepare features and target
    features = ['molecular_weight', 'logp', 'polar_surface_area', 
                'num_rotatable_bonds', 'num_aromatic_rings',
                'num_hydrogen_donors', 'num_hydrogen_acceptors']
    
    X = df[features]
    y = df['efficacy_score']
    
    # Split data
    print("2. Splitting data into train/test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Scale features
    print("3. Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    print("4. Training Random Forest model...")
    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        max_depth=10
    )
    
    model.fit(X_train_scaled, y_train)
    
    # Make predictions
    print("5. Evaluating model performance...")
    y_pred = model.predict(X_test_scaled)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    print(f"\n=== Model Performance ===")
    print(f"R² Score: {r2:.3f}")
    print(f"RMSE: {rmse:.3f}")
    print(f"Mean Squared Error: {mse:.3f}")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': features,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\n=== Feature Importance ===")
    for _, row in feature_importance.iterrows():
        print(f"{row['feature']}: {row['importance']:.3f}")
    
    # Save model and scaler
    print("\n6. Saving trained model...")
    os.makedirs('models', exist_ok=True)
    
    with open('models/drug_efficacy_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    with open('models/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    print("   Model saved to: models/drug_efficacy_model.pkl")
    print("   Scaler saved to: models/scaler.pkl")
    
    print("\n*** Model training completed successfully! ***")
    return model, scaler, feature_importance

if __name__ == "__main__":
    model, scaler, importance = train_drug_efficacy_model()