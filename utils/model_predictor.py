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

# --- Function for Prediction and Reporting ---
def generate_prediction_and_report(features, model):
    """
    Uses the trained model to predict and generate a report.
    """
    print("\n--- Generating Patient Report ---")

    input_data = pd.DataFrame([features], columns=TRAIT_COLUMNS)
    
    prediction = model.predict(input_data)
    prediction_proba = model.predict_proba(input_data)
    
    print("-" * 30)
    print("Detected Behavioral Traits:")
    for trait, score in features.items():
        print(f"  - {trait:<40}: {score:.2f}")
    
    print("-" * 30)
    print("Autism Likelihood Prediction:")
    if prediction[0] == 1:
        print("  Prediction: AUTISTIC")
        print(f"  Confidence: {prediction_proba[0][1]:.2%} likelihood of autism")
    else:
        print("  Prediction: NOT AUTISTIC")
        print(f"  Confidence: {prediction_proba[0][0]:.2%} likelihood of not having autism")
        
