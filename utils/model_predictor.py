import pandas as pd
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from config.config import TRAIT_COLUMNS, MODEL_PATH

# --- Functions for Model Management ---
def train_and_save_model():
    """Trains a simple Random Forest model and saves it."""
    print("Pre-trained model not found. Training a simple model for demonstration...")
    
    # Create a dummy training DataFrame
    dummy_data = {
        'Absence or Avoidance of Eye Contact': [0.9, 0.1, 0.8, 0.2, 0.9, 0.1],
        'Aggressive Behavior': [0.5, 0.1, 0.7, 0.0, 0.6, 0.2],
        'Hyper- or Hyporeactivity to Sensory Input': [0.8, 0.2, 0.9, 0.1, 0.7, 0.2],
        'Non-Responsiveness to Verbal Interaction': [0.7, 0.3, 0.8, 0.2, 0.6, 0.1],
        'Non-Typical Language': [0.9, 0.1, 0.8, 0.3, 0.7, 0.1],
        'Object Lining-Up': [0.6, 0.1, 0.7, 0.0, 0.5, 0.0],
        'Self-Hitting or Self-Injurious Behavior': [0.8, 0.1, 0.9, 0.0, 0.7, 0.0],
        'Self-Spinning or Spinning Objects': [0.9, 0.0, 0.8, 0.1, 0.9, 0.0],
        'Upper Limb Stereotypies': [0.7, 0.2, 0.8, 0.1, 0.6, 0.1],
        'is_autistic': [1, 0, 1, 0, 1, 0]
    }
    dummy_df = pd.DataFrame(dummy_data)
    
    X = dummy_df[TRAIT_COLUMNS]
    y = dummy_df['is_autistic']
    
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
    
    predictor_model = RandomForestClassifier(random_state=42)
    predictor_model.fit(X_train, y_train)
    
    joblib.dump(predictor_model, MODEL_PATH)
    print(f"Model trained and saved to '{MODEL_PATH}'.")
    return predictor_model

def load_model():
    """Loads a saved model from disk."""
    if os.path.exists(MODEL_PATH):
        print(f"Loading pre-trained model from '{MODEL_PATH}'...")
        return joblib.load(MODEL_PATH)
    else:
        return train_and_save_model()
        

def generate_prediction_and_report(features, model):
    """
    Uses the trained model to predict and generate a report dictionary.
    This version RETURNS a dictionary instead of printing.
    """
    print("\n--- Generating Patient Report ---")
    
    # It's crucial that the feature order matches the model's training order.
    # Assuming TRAIT_COLUMNS is accessible or defined as below.
    TRAIT_COLUMNS = list(features.keys()) 
    input_data = pd.DataFrame([features], columns=TRAIT_COLUMNS)
    
    prediction = model.predict(input_data)
    prediction_proba = model.predict_proba(input_data)
    
    if prediction[0] == 1:
        result = {
            "prediction": "AUTISTIC",
            "confidence": f"{prediction_proba[0][1]:.2%}",
            "likelihood_score": prediction_proba[0][1]
        }
    else:
        result = {
            "prediction": "NOT AUTISTIC",
            "confidence": f"{prediction_proba[0][0]:.2%}",
            "likelihood_score": prediction_proba[0][0]
        }
        
    # Combine trait scores and the final prediction into one report
    report = {
        "detected_traits": features,
        "final_prediction": result
    }
    
    return report
