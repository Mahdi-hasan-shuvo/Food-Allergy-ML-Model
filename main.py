import joblib
import pandas as pd
import numpy as np

# Load the saved model
def load_model(model_path=r'/content/food_allergy_model.pkl'):
    """Load the saved model, scaler, and label encoders"""
    try:
        saved_data = joblib.load(model_path)
        return saved_data['model'], saved_data['scaler'], saved_data['label_encoders']
    except FileNotFoundError:
        print(f"Model file {model_path} not found.")
        return None, None, None
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None, None

# Load the model
model, scaler, label_encoders = load_model()

if model is not None:
    print("Model loaded successfully!")
    
    # Check what categories the model was trained on
    print("\nCategories the model was trained on:")
    for column, encoder in label_encoders.items():
        print(f"{column}: {list(encoder.classes_)}")
    
    # Function to safely preprocess new data
    def safe_predict_allergy(new_data, model, scaler, label_encoders):
        """
        Predict allergy for new data with error handling for unseen labels
        """
        # Create DataFrame from new data
        new_df = pd.DataFrame([new_data])
        
        # Preprocess the new data using the same label encoders
        for col in label_encoders:
            if col in new_df.columns:
                # Check if the value exists in the encoder's classes
                if new_df[col].iloc[0] not in label_encoders[col].classes_:
                    print(f"Warning: '{new_df[col].iloc[0]}' not seen in training for {col}.")
                    # Use the most common category as fallback
                    most_common = label_encoders[col].classes_[0]
                    print(f"Using '{most_common}' as fallback.")
                    new_df[col] = label_encoders[col].transform([most_common])[0]
                else:
                    new_df[col] = label_encoders[col].transform(new_df[col].astype(str))
        
        # Scale the features
        new_scaled = scaler.transform(new_df)
        
        # Make prediction
        prediction = model.predict(new_scaled)
        probability = model.predict_proba(new_scaled)[0] if hasattr(model, 'predict_proba') else None
        
        return prediction[0], probability

    # Test with data that uses known categories
    print("\nTesting with known categories:")
    known_categories_patient = {
        'Age': 30,
        'Gender': 'Male',  # Should be in training data
        'Family_History': 'No',  # Should be in training data
        'Previous_Reaction': 'Moderate',  # Should be in training data
        'Symptoms': 'Swelling',  # Should be in training data
        'Food_Type': 'Nuts',  # Should be in training data
        'Food_Frequency': 10,
        'Medical_Conditions': 'Asthma',  # Should be in training data
        'IgE_Levels': 500.0,
        'Severity_Score': 7
    }
    
    prediction, probability = safe_predict_allergy(known_categories_patient, model, scaler, label_encoders)
    print(f"Prediction: {'Allergic' if prediction == 1 else 'Not Allergic'}")
    if probability is not None:
        print(f"Probability: {probability[1]:.4f}")
    
    # Test with data that includes unknown categories
    print("\nTesting with unknown category (should show warning):")
    unknown_category_patient = {
        'Age': 25,
        'Gender': 'Male',
        'Family_History': 'No',
        'Previous_Reaction': 'Severe',
        'Symptoms': 'Hives',  # This is the unknown category that caused the error
        'Food_Type': 'Shellfish',
        'Food_Frequency': 5,
        'Medical_Conditions': 'None',
        'IgE_Levels': 650.0,
        'Severity_Score': 8
    }
    
    prediction, probability = safe_predict_allergy(unknown_category_patient, model, scaler, label_encoders)
    print(f"Prediction: {'Allergic' if prediction == 1 else 'Not Allergic'}")
    if probability is not None:
        print(f"Probability: {probability[1]:.4f}")

else:
    print("Failed to load model. Please check if the model file exists.")
