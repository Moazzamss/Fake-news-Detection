from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np
import os

app = Flask(__name__)
CORS(app, origins="*")

# Global variables for models
vectorizer = None
lr_model = None
dt_model = None
nb_model = None
rf_model = None
models_loaded = False

def load_models():
    global vectorizer, lr_model, dt_model, nb_model, rf_model, models_loaded
    
    try:
        model_path = os.path.join(os.path.dirname(__file__), 'models')
        
        print("🔄 Loading vectorizer...")
        with open(os.path.join(model_path, 'vectorizer.pkl'), 'rb') as f:
            vectorizer = pickle.load(f)
        print("✅ Vectorizer loaded")
        
        print("🔄 Loading LR model...")
        with open(os.path.join(model_path, 'lr_model.pkl'), 'rb') as f:
            lr_model = pickle.load(f)
        print("✅ LR model loaded")
        
        print("🔄 Loading DT model...")
        with open(os.path.join(model_path, 'dt_model.pkl'), 'rb') as f:
            dt_model = pickle.load(f)
        print("✅ DT model loaded")
        
        print("🔄 Loading NB model...")
        with open(os.path.join(model_path, 'nb_model.pkl'), 'rb') as f:
            nb_model = pickle.load(f)
        print("✅ NB model loaded")
        
        print("🔄 Loading RF model...")
        with open(os.path.join(model_path, 'rf_model.pkl'), 'rb') as f:
            rf_model = pickle.load(f)
        print("✅ RF model loaded")
        
        models_loaded = True
        print("🎉 ALL MODELS LOADED SUCCESSFULLY!")
        return True
        
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        import traceback
        traceback.print_exc()
        return False

# Load models on startup
print("=" * 50)
print("🚀 STARTING FLASK APP...")
print("=" * 50)
load_models()

@app.route('/')
def home():
    return jsonify({
        "message": "Flask API is ready",
        "status": "Backend is running!",
        "models_loaded": models_loaded,
        "endpoints": ["/predict", "/api/predict", "/api/health"]
    })

@app.route('/api/health')
def health():
    return jsonify({
        "status": "healthy" if models_loaded else "unhealthy",
        "models_loaded": models_loaded
    })

@app.route('/predict', methods=['POST', 'OPTIONS'])
@app.route('/api/predict', methods=['POST', 'OPTIONS'])
def predict():
    if request.method == 'OPTIONS':
        response = jsonify({'status': 'ok'})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
        response.headers.add('Access-Control-Allow-Methods', 'POST')
        return response, 200
    
    if not models_loaded:
        return jsonify({'error': 'Models not loaded'}), 500
    
    try:
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({'error': 'No text provided'}), 400
        
        text = data['text']
        
        if not text or len(text.strip()) == 0:
            return jsonify({'error': 'Text is empty'}), 400
        
        # Vectorize the input text
        text_vectorized = vectorizer.transform([text])
        
        # Get predictions from all models
        lr_pred = lr_model.predict(text_vectorized)[0]
        lr_proba = lr_model.predict_proba(text_vectorized)[0]
        
        dt_pred = dt_model.predict(text_vectorized)[0]
        dt_proba = dt_model.predict_proba(text_vectorized)[0]
        
        nb_pred = nb_model.predict(text_vectorized)[0]
        nb_proba = nb_model.predict_proba(text_vectorized)[0]
        
        rf_pred = rf_model.predict(text_vectorized)[0]
        rf_proba = rf_model.predict_proba(text_vectorized)[0]
        
        # Count how many models predict FAKE (1) vs REAL (0)
        predictions = [lr_pred, dt_pred, nb_pred, rf_pred]
        fake_count = predictions.count(1)  # Count how many predict FAKE (1)
        real_count = predictions.count(0)  # Count how many predict REAL (0)
        
        # If 2 or more models say FAKE, final prediction is FAKE
        # Otherwise REAL
        if fake_count >= 2:
            final_prediction = 1  # FAKE
        else:
            final_prediction = 0  # REAL
        
        # Calculate average confidence for the final prediction
        # Only average the models that voted for the final prediction
        confidence_values = []
        if final_prediction == 1:  # If FAKE won
            if lr_pred == 1:
                confidence_values.append(lr_proba[1])
            if dt_pred == 1:
                confidence_values.append(dt_proba[1])
            if nb_pred == 1:
                confidence_values.append(nb_proba[1])
            if rf_pred == 1:
                confidence_values.append(rf_proba[1])
        else:  # If REAL won
            if lr_pred == 0:
                confidence_values.append(lr_proba[0])
            if dt_pred == 0:
                confidence_values.append(dt_proba[0])
            if nb_pred == 0:
                confidence_values.append(nb_proba[0])
            if rf_pred == 0:
                confidence_values.append(rf_proba[0])
        
        # Average confidence of models that agree with final prediction
        avg_confidence = np.mean(confidence_values) * 100 if confidence_values else 0
        
        result = {
            'prediction': 'FAKE' if final_prediction == 1 else 'REAL',
            'confidence': round(avg_confidence, 2),
            'model_predictions': {
                'logistic_regression': 'FAKE' if lr_pred == 1 else 'REAL',
                'decision_tree': 'FAKE' if dt_pred == 1 else 'REAL',
                'naive_bayes': 'FAKE' if nb_pred == 1 else 'REAL',
                'random_forest': 'FAKE' if rf_pred == 1 else 'REAL'
            },
            'individual_confidences': {
                'logistic_regression': round(lr_proba[int(lr_pred)] * 100, 2),
                'decision_tree': round(dt_proba[int(dt_pred)] * 100, 2),
                'naive_bayes': round(nb_proba[int(nb_pred)] * 100, 2),
                'random_forest': round(rf_proba[int(rf_pred)] * 100, 2)
            }
        }
        
        return jsonify(result)
    
    except Exception as e:
        print(f"❌ Error in prediction: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"🌐 Starting server on port {port}...")
    app.run(host='0.0.0.0', port=port, debug=False)