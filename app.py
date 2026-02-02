from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np
import pandas as pd
import os

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

print("Loading models...")
try:
    with open('models/rf_model.pkl', 'rb') as f:
        rf = pickle.load(f)
    with open('models/dt_model.pkl', 'rb') as f:
        dt = pickle.load(f)
    with open('models/lr_model.pkl', 'rb') as f:
        lr = pickle.load(f)
    with open('models/nb_model.pkl', 'rb') as f:
        nb = pickle.load(f)
    with open('models/vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    print("✅ Models loaded!")
except Exception as e:
    print(f"❌ Error: {e}")

def majority_voting(*predictions):
    predictions_flat = np.concatenate(predictions)
    fake_count = np.sum(predictions_flat == 1)
    return 1 if fake_count > 1 else 0

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        'status': 'Backend is running!',
        'message': 'Flask API is ready',
        'endpoints': ['/predict']
    })

@app.route('/predict', methods=['POST', 'OPTIONS'])
def predict():
    if request.method == 'OPTIONS':
        return '', 204
    
    try:
        text = request.form.get('article')
        if not text:
            return jsonify({'error': 'No article', 'result': 'ERROR'}), 400

        cleaned_text = text.lower()
        vectorized_text = vectorizer.transform([cleaned_text])

        rf_pred = rf.predict(vectorized_text)
        dt_pred = dt.predict(vectorized_text)
        lr_pred = lr.predict(vectorized_text)
        nb_pred = nb.predict(vectorized_text)

        predictions = np.array([rf_pred, dt_pred, lr_pred, nb_pred])
        final_pred = majority_voting(*predictions)
        result = 'FAKE' if final_pred == 1 else 'REAL'

        model_predictions = [
            {'name': 'Random Forest', 'prediction': int(rf_pred[0])},
            {'name': 'Decision Tree', 'prediction': int(dt_pred[0])},
            {'name': 'Logistic Regression', 'prediction': int(lr_pred[0])},
            {'name': 'Naive Bayes', 'prediction': int(nb_pred[0])}
        ]

        return jsonify({'result': result, 'model_predictions': model_predictions})
    
    except Exception as e:
        return jsonify({'error': str(e), 'result': 'ERROR'}), 500

if __name__ == '__main__':
    # CHANGED: Use environment port for Render, default to 5000 for local
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)