"""
HIV Activity Prediction API
"""
import pickle
import pandas as pd
import numpy as np

class HIVActivityPredictor:
    def __init__(self):
        """Load the trained HIV activity classification model"""
        try:
            with open('models/hiv_activity_model.pkl', 'rb') as f:
                self.model = pickle.load(f)
            
            with open('models/scaler.pkl', 'rb') as f:
                self.scaler = pickle.load(f)
            
            with open('models/label_encoder.pkl', 'rb') as f:
                self.label_encoder = pickle.load(f)
            
            with open('models/features.pkl', 'rb') as f:
                self.features = pickle.load(f)
            
            self.classes = self.label_encoder.classes_
            print(f"HIV Activity Model loaded successfully!")
            print(f"Features: {self.features}")
            print(f"Classes: {list(self.classes)}")
            
        except FileNotFoundError as e:
            print(f"ERROR: Model files not found - {e}")
            print("Please run train_model.py first to train the HIV activity model.")
            raise
    
    def predict_single_compound(self, compound_properties):
        """Predict HIV activity for a single compound"""
        # Convert to DataFrame
        df = pd.DataFrame([compound_properties])
        
        # Ensure all required features are present
        for feature in self.features:
            if feature not in df.columns:
                # Provide default values for missing features
                if feature == 'Log10EC50':
                    df[feature] = -5.0  # Default EC50
                elif feature == 'IC50':
                    df[feature] = 50.0  # Default IC50
                elif feature == 'Log_SI':
                    df[feature] = 1.0   # Default SI
                else:
                    df[feature] = 0.0   # Default for molecular features
        
        # Ensure correct feature order
        df = df[self.features]
        
        # Handle missing values
        df = df.fillna(df.median())
        
        # Scale features
        df_scaled = self.scaler.transform(df)
        
        # Make prediction
        prediction = self.model.predict(df_scaled)[0]
        probabilities = self.model.predict_proba(df_scaled)[0]
        
        # Get prediction confidence
        max_prob = np.max(probabilities)
        
        # Convert prediction back to class name
        predicted_class = self.label_encoder.inverse_transform([prediction])[0]
        
        # Create probability dictionary
        class_probs = {}
        for i, class_name in enumerate(self.classes):
            class_probs[class_name] = round(probabilities[i], 3)
        
        return {
            'predicted_activity': predicted_class,
            'confidence': round(max_prob, 3),
            'class_probabilities': class_probs,
            'interpretation': self.interpret_activity(predicted_class, max_prob)
        }
    
    def predict_batch(self, compounds_df):
        """Predict HIV activity for multiple compounds"""
        results = []
        
        for idx, row in compounds_df.iterrows():
            compound_dict = row.to_dict()
            if 'name' in compound_dict:
                name = compound_dict.pop('name')
            else:
                name = f"Compound_{idx}"
            
            try:
                result = self.predict_single_compound(compound_dict)
                result['compound_name'] = name
                results.append(result)
            except Exception as e:
                print(f"Error predicting {name}: {e}")
                results.append({
                    'compound_name': name,
                    'predicted_activity': 'ERROR',
                    'confidence': 0.0,
                    'interpretation': f'Prediction failed: {e}'
                })
        
        return results
    
    def interpret_activity(self, activity, confidence):
        """Interpret the predicted HIV activity"""
        base_interpretation = {
            'CA': 'Confirmed Active - Strong anti-HIV activity',
            'CM': 'Confirmed Moderately Active - Moderate anti-HIV activity', 
            'CI': 'Confirmed Inactive - No significant anti-HIV activity'
        }
        
        confidence_level = "High" if confidence >= 0.8 else "Medium" if confidence >= 0.6 else "Low"
        
        return f"{base_interpretation.get(activity, 'Unknown activity')} (Confidence: {confidence_level})"

def demo_hiv_predictions():
    """Demo the HIV activity prediction system"""
    print("=== HIV Activity Prediction Demo ===")
    
    # Load predictor
    predictor = HIVActivityPredictor()
    
    # Test compounds with different molecular properties
    test_compounds = [
        {
            'name': 'Compound A (Drug-like)',
            'molecular_weight': 350,
            'logp': 2.8,  # Good lipophilicity
            'polar_surface_area': 75,
            'num_rotatable_bonds': 4,
            'num_aromatic_rings': 2,
            'num_hydrogen_donors': 2,
            'num_hydrogen_acceptors': 4,
            'num_heavy_atoms': 25,
            'Log10EC50': -6.5,  # Good efficacy
            'IC50': 100,        # Low toxicity
        },
        {
            'name': 'Compound B (Poor Properties)',
            'molecular_weight': 600,  # Too large
            'logp': -1.0,            # Poor lipophilicity
            'polar_surface_area': 150, # Too polar
            'num_rotatable_bonds': 12,
            'num_aromatic_rings': 1,
            'num_hydrogen_donors': 6,
            'num_hydrogen_acceptors': 10,
            'num_heavy_atoms': 45,
            'Log10EC50': -4.0,  # Poor efficacy
            'IC50': 10,         # High toxicity
        },
        {
            'name': 'Compound C (Balanced)',
            'molecular_weight': 280,
            'logp': 1.8,
            'polar_surface_area': 85,
            'num_rotatable_bonds': 3,
            'num_aromatic_rings': 2,
            'num_hydrogen_donors': 1,
            'num_hydrogen_acceptors': 3,
            'num_heavy_atoms': 20,
            'Log10EC50': -5.8,  # Moderate efficacy
            'IC50': 50,         # Moderate toxicity
        }
    ]
    
    print(f"\nTesting {len(test_compounds)} compounds...")
    
    for compound in test_compounds:
        name = compound.pop('name')
        result = predictor.predict_single_compound(compound)
        
        print(f"\n--- {name} ---")
        print(f"Predicted Activity: {result['predicted_activity']}")
        print(f"Confidence: {result['confidence']}")
        print(f"Interpretation: {result['interpretation']}")
        print("Class Probabilities:")
        for class_name, prob in result['class_probabilities'].items():
            print(f"  {class_name}: {prob}")

def predict_from_csv(csv_file):
    """Predict HIV activity for compounds from a CSV file"""
    print(f"=== Batch Prediction from {csv_file} ===")
    
    try:
        # Load compounds from CSV
        compounds_df = pd.read_csv(csv_file)
        print(f"Loaded {len(compounds_df)} compounds from {csv_file}")
        
        # Load predictor
        predictor = HIVActivityPredictor()
        
        # Make predictions
        results = predictor.predict_batch(compounds_df)
        
        # Save results
        results_df = pd.DataFrame(results)
        output_file = csv_file.replace('.csv', '_predictions.csv')
        results_df.to_csv(output_file, index=False)
        
        print(f"\nPredictions saved to: {output_file}")
        
        # Show summary
        activity_counts = results_df['predicted_activity'].value_counts()
        print(f"\nPrediction Summary:")
        for activity, count in activity_counts.items():
            print(f"  {activity}: {count}")
        
        return results_df
        
    except FileNotFoundError:
        print(f"Error: Could not find {csv_file}")
    except Exception as e:
        print(f"Error processing {csv_file}: {e}")

if __name__ == "__main__":
    # Run demo predictions
    demo_hiv_predictions()
    
    # Uncomment below to predict from a CSV file
    # predict_from_csv('compounds.csv')