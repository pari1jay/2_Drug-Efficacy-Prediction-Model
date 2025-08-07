"""
Drug Efficacy Prediction API
"""
import pickle
import pandas as pd
import numpy as np

class DrugEfficacyPredictor:
    def __init__(self):
        """Load the trained model and scaler"""
        try:
            with open('models/drug_efficacy_model.pkl', 'rb') as f:
                self.model = pickle.load(f)
            
            with open('models/scaler.pkl', 'rb') as f:
                self.scaler = pickle.load(f)
            
            self.features = ['molecular_weight', 'logp', 'polar_surface_area', 
                           'num_rotatable_bonds', 'num_aromatic_rings',
                           'num_hydrogen_donors', 'num_hydrogen_acceptors']
            
            print("Model loaded successfully!")
            
        except FileNotFoundError:
            print("ERROR: Model files not found. Please run train_model.py first.")
            raise
    
    def predict_single_drug(self, drug_properties):
        """Predict efficacy for a single drug"""
        # Convert to DataFrame
        df = pd.DataFrame([drug_properties])
        
        # Ensure correct feature order
        df = df[self.features]
        
        # Scale features
        df_scaled = self.scaler.transform(df)
        
        # Make prediction
        efficacy_score = self.model.predict(df_scaled)[0]
        
        # Get prediction confidence (using random forest variance)
        predictions = []
        for estimator in self.model.estimators_:
            pred = estimator.predict(df_scaled)[0]
            predictions.append(pred)
        
        confidence = 1.0 - np.std(predictions)  # Higher std = lower confidence
        
        return {
            'efficacy_score': round(efficacy_score, 3),
            'confidence': round(confidence, 3),
            'interpretation': self.interpret_score(efficacy_score)
        }
    
    def interpret_score(self, score):
        """Interpret the efficacy score"""
        if score >= 0.8:
            return "Highly Effective"
        elif score >= 0.6:
            return "Moderately Effective" 
        elif score >= 0.4:
            return "Low Effectiveness"
        else:
            return "Likely Ineffective"

def demo_predictions():
    """Demo the prediction system"""
    print("=== Drug Efficacy Prediction Demo ===")
    
    # Load predictor
    predictor = DrugEfficacyPredictor()
    
    # Test drugs with different properties
    test_drugs = [
        {
            'name': 'Drug A (Optimized)',
            'molecular_weight': 300,
            'logp': 2.5,  # Optimal LogP
            'polar_surface_area': 80,
            'num_rotatable_bonds': 5,
            'num_aromatic_rings': 2,
            'num_hydrogen_donors': 2,
            'num_hydrogen_acceptors': 4
        },
        {
            'name': 'Drug B (Poor LogP)',
            'molecular_weight': 280,
            'logp': 0.5,  # Poor LogP
            'polar_surface_area': 90,
            'num_rotatable_bonds': 3,
            'num_aromatic_rings': 1,
            'num_hydrogen_donors': 3,
            'num_hydrogen_acceptors': 5
        },
        {
            'name': 'Drug C (Large Molecule)',
            'molecular_weight': 500,  # Very large
            'logp': 2.0,
            'polar_surface_area': 120,
            'num_rotatable_bonds': 8,
            'num_aromatic_rings': 3,
            'num_hydrogen_donors': 4,
            'num_hydrogen_acceptors': 6
        }
    ]
    
    for drug in test_drugs:
        name = drug.pop('name')
        result = predictor.predict_single_drug(drug)
        
        print(f"\n--- {name} ---")
        print(f"Predicted Efficacy: {result['efficacy_score']} ({result['interpretation']})")
        print(f"Confidence: {result['confidence']}")

if __name__ == "__main__":
    demo_predictions()