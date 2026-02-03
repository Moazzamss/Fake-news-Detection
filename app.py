from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np
import os

app = Flask(__name__)

# Enable CORS for all origins (important for Vercel)
CORS(app, origins="*")

# Load models
try:
    model_path = os.path.join(os.path.dirname(__file__), 'models')
    
    with open(os.path.join(model_path, 'vectorizer.pkl'), 'rb') as f:
        vectorizer = pickle.load(f)
    
    with open(os.path.join(model_path, 'lr_model.pkl'), 'rb') as f:
        lr_model = pickle.load(f)
    
    with open(os.path.join(model_path, 'dt_model.pkl'), 'rb') as f:
        dt_model = pickle.load(f)
    
    with open(os.path.join(model_path, 'gb_model.pkl'), 'rb') as f:
        gb_model = pickle.load(f)
    
    with open(os.path.join(model_path, 'rf_model.pkl'), 'rb') as f:
        rf_model = pickle.load(f)
    
    print("✅ All models loaded successfully!")
except Exception as e:
    print(f"❌ Error loading models: {e}")

@app.route('/')
def home():
    return jsonify({
        "message": "Flask API is ready",
        "status": "Backend is running!",
        "endpoints": ["/predict", "/api/predict"]
    })

@app.route('/api/health')
def health():
    return jsonify({
        "status": "healthy",
        "models_loaded": True
    })

# Support both /predict and /api/predict
@app.route('/predict', methods=['POST', 'OPTIONS'])
@app.route('/api/predict', methods=['POST', 'OPTIONS'])
def predict():
    # Handle preflight OPTIONS request for CORS
    if request.method == 'OPTIONS':
        response = jsonify({'status': 'ok'})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
        response.headers.add('Access-Control-Allow-Methods', 'POST')
        return response, 200
    
    try:
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({
                'error': 'No text provided. Please send JSON with "text" field.'
            }), 400
        
        text = data['text']
        
        if not text or len(text.strip()) == 0:
            return jsonify({
                'error': 'Text is empty. Please provide some news text to analyze.'
            }), 400
        
        # Vectorize the input text
        text_vectorized = vectorizer.transform([text])
        
        # Get predictions from all models
        lr_pred = lr_model.predict(text_vectorized)[0]
        lr_proba = lr_model.predict_proba(text_vectorized)[0]
        
        dt_pred = dt_model.predict(text_vectorized)[0]
        dt_proba = dt_model.predict_proba(text_vectorized)[0]
        
        gb_pred = gb_model.predict(text_vectorized)[0]
        gb_proba = gb_model.predict_proba(text_vectorized)[0]
        
        rf_pred = rf_model.predict(text_vectorized)[0]
        rf_proba = rf_model.predict_proba(text_vectorized)[0]
        
        # Ensemble voting
        predictions = [lr_pred, dt_pred, gb_pred, rf_pred]
        final_prediction = max(set(predictions), key=predictions.count)
        
        # Average confidence
        avg_confidence = np.mean([
            lr_proba[int(final_prediction)],
            dt_proba[int(final_prediction)],
            gb_proba[int(final_prediction)],
            rf_proba[int(final_prediction)]
        ]) * 100
        
        result = {
            'prediction': 'REAL' if final_prediction == 1 else 'FAKE',
            'confidence': round(avg_confidence, 2),
            'model_predictions': {
                'logistic_regression': 'REAL' if lr_pred == 1 else 'FAKE',
                'decision_tree': 'REAL' if dt_pred == 1 else 'FAKE',
                'gradient_boosting': 'REAL' if gb_pred == 1 else 'FAKE',
                'random_forest': 'REAL' if rf_pred == 1 else 'FAKE'
            },
            'individual_confidences': {
                'logistic_regression': round(lr_proba[int(lr_pred)] * 100, 2),
                'decision_tree': round(dt_proba[int(dt_pred)] * 100, 2),
                'gradient_boosting': round(gb_proba[int(gb_pred)] * 100, 2),
                'random_forest': round(rf_proba[int(rf_pred)] * 100, 2)
            }
        }
        
        return jsonify(result)
    
    except Exception as e:
        print(f"❌ Error in prediction: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'error': f'Prediction failed: {str(e)}'
        }), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)