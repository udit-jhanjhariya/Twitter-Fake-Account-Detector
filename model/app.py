from flask import Flask, render_template, request, jsonify
import pandas as pd
from joblib import load
import gender_guesser.detector as gender
from waitress import serve
import os

app = Flask(__name__)

# Global model variable
model = None

def load_model():
    global model
    try:
        # Get the absolute path to the model file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_dir, 'twitter_fake_account_detector.joblib')
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
        
        model = load(model_path)
        print("Model loaded successfully!")
        print(f"Model type: {type(model)}")
        print(f"Model parameters: {model.get_params()}")
        print(f"Model feature names: {model.feature_names_in_}")
        print(f"Model attributes: {dir(model)}")
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        raise e

# Load the model when the app starts
load_model()

# Initialize gender detector
gender_detector = gender.Detector()

@app.route('/')
def home():
    print("Received request for home page")
    try:
        return render_template('index.html')
    except Exception as e:
        print(f"Error rendering template: {str(e)}")
        return f"Error: {str(e)}"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        print("Predict function called")
        if model is None:
            raise ValueError("Model is not loaded")
            
        print(f"Model in predict: {model}")
        
        # Get data from the form
        data = {
            'Sex Code': [int(request.form.get('gender', 2))],
            'statuses_count': [int(request.form.get('tweets', 0))],
            'followers_count': [int(request.form.get('followers', 0))],
            'friends_count': [int(request.form.get('following', 0))],
            'favourites_count': [int(request.form.get('favorites', 0))],
            'listed_count': [int(request.form.get('listed', 0))],
            'lang_code': [int(request.form.get('language', 1))]
        }

        print(f"Data received: {data}")

        # Convert to DataFrame
        df = pd.DataFrame(data)
        print(f"DataFrame created: {df}")

        # Use the exact feature names and order from the model
        features = model.feature_names_in_.tolist()
        print(f"Using model features in correct order: {features}")

        # Make prediction
        X = df[features]
        print(f"Features selected: {X}")
        
        prediction = model.predict(X)
        probability = model.predict_proba(X)
        
        print(f"Prediction made: {prediction}")
        print(f"Probabilities: {probability}")

        result = {
            'prediction': 'Real Account' if prediction[0] == 1 else 'Fake Account',
            'confidence': f"{probability[0][prediction[0]]*100:.2f}%",
            'fake_probability': f"{probability[0][0]*100:.2f}%",
            'real_probability': f"{probability[0][1]*100:.2f}%"
        }

        return jsonify(result)

    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        return jsonify({'error': str(e)}), 400


if __name__ == '__main__':
    print("\n" + "="*50)
    print("Starting Twitter Fake Profile Detection Server")
    print("="*50)
    print("\nAccess the application at: http://127.0.0.1:8080")
    print("\nPress Ctrl+C to stop the server")
    print("="*50 + "\n")
    serve(app, host='127.0.0.1', port=8080) 



