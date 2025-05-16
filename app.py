import numpy as np
import pickle
from flask import Flask, render_template, request, redirect, url_for
from sklearn.preprocessing import StandardScaler # This import was present, ensuring it's used.
from tensorflow.keras.models import load_model
import os
import traceback

app = Flask(__name__)

# Global variables to hold the loaded model and scaler
model = None
scaler = None

def load_assets():
    """Loads the Keras model and the scaler from local files."""
    global model, scaler
    print("Attempting to load model and scaler assets from local files...")

    # Define local paths for model and scaler
    # Ensure these files are in the same directory as app.py or provide correct path
    local_model_path = 'my_model.h5'
    local_scaler_path = 'scaler.pkl'

    try:
        if not os.path.exists(local_model_path):
            print(f"CRITICAL ERROR: Model file not found at {local_model_path}")
            return False
        model = load_model(local_model_path)
        print(f"Keras model loaded successfully from {local_model_path}.")
    except Exception as e:
        print(f"Error loading Keras model from {local_model_path}: {e}")
        traceback.print_exc()
        return False

    try:
        if not os.path.exists(local_scaler_path):
            print(f"CRITICAL ERROR: Scaler file not found at {local_scaler_path}")
            if model: model = None # Unload model if scaler is missing
            return False
        with open(local_scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        print(f"Scaler loaded successfully from {local_scaler_path}.")
    except Exception as e:
        print(f"Error loading scaler from {local_scaler_path}: {e}")
        traceback.print_exc()
        if model: model = None # Unload model if scaler loading fails
        return False

    return True # Both loaded successfully

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/form')
def form():
    return render_template('form.html')

@app.route('/results', methods=['POST'])
def results():
    if model is None or scaler is None:
        print("ERROR in /results: Model or scaler is not loaded. Attempting to reload or redirecting.")
        # Optionally, try to reload assets here if appropriate for your app's logic
        # or simply redirect. For simplicity, redirecting.
        if not load_assets(): # Attempt to reload if not loaded
             print("Failed to reload assets. Redirecting to form.")
             return redirect(url_for('form'))
        # If load_assets was successful, proceed, otherwise it would have returned False above.

    try:
        age = int(request.form.get('age', 0))
        gender_str = request.form.get('gender', '').strip().lower()

        if gender_str == 'male':
            gender_encoded = 1
        elif gender_str == 'female':
            gender_encoded = 2
        else:
            print(f"Warning: Invalid gender input '{request.form.get('gender')}', defaulting gender_encoded to 0.")
            gender_encoded = 0 # Default or handle error appropriately

        features_names = [
            'air_pollution', 'alcohol_use', 'dust_allergy', 'occupational_hazards',
            'genetic_risk', 'chronic_lung_disease', 'balanced_diet', 'obesity',
            'smoking', 'passive_smoker', 'chest_pain', 'coughing_blood', 'fatigue',
            'weight_loss', 'shortness_of_breath', 'wheezing', 'swallowing_difficulty',
            'clubbing', 'frequent_cold', 'dry_cough', 'snoring'
        ]

        form_features = []
        for fname in features_names:
            val_str = request.form.get(fname)
            try:
                # Ensure value is treated as int, default to 0 if empty or invalid
                form_features.append(int(val_str) if val_str and val_str.strip().isdigit() else 0)
            except ValueError:
                print(f"Warning: Invalid non-integer value for {fname} ('{val_str}'), defaulting to 0.")
                form_features.append(0)

        # Data construction as per your train.py and test.py (Age, Gender, then 21 features)
        data_to_predict = np.array([age, gender_encoded] + form_features).reshape(1, -1)
        
        expected_features = 23 # X.shape[1] from train.py (File [8])
        if data_to_predict.shape[1] != expected_features:
            print(f"CRITICAL ERROR: Input data has {data_to_predict.shape[1]} features, but model expects {expected_features}.")
            print("Data received:", data_to_predict.tolist()) # Log the problematic data
            # Consider flashing a message to the user here instead of just redirecting
            return redirect(url_for('form'))

        data_scaled = scaler.transform(data_to_predict)
        pred_probs = model.predict(data_scaled)[0]
        
        predicted_index = np.argmax(pred_probs)
        class_map = {0: 'Low', 1: 'Medium', 2: 'High'} # From train.py (File [8]) / test.py (File [7])
        predicted_label = class_map.get(predicted_index, 'Unknown')

        return render_template(
            'results.html',
            outcome=predicted_label,
            probabilities=pred_probs.tolist() # Pass probabilities to template
        )

    except Exception as e:
        print(f"Error during prediction in /results route: {e}")
        traceback.print_exc()
        # Consider flashing a message to the user here instead of just redirecting
        return redirect(url_for('form'))

if __name__ == '__main__':
    if load_assets():
        print("Application starting: Model and scaler loaded successfully.")
        # Use PORT environment variable if available (for Heroku, Render, etc.)
        port = int(os.environ.get("PORT", 5000))
        app.run(host='0.0.0.0', port=port, debug=False) # Set debug=False for production
    else:
        print("CRITICAL ERROR: Application failed to start because model and/or scaler could not be loaded.")
        print("Ensure 'my_model.h5' and 'scaler.pkl' are in the same directory as app.py.")

