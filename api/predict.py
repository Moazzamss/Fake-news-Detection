# api/predict.py

from flask import Flask, jsonify, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the model
with open('models/rf_model.pkl', 'rb') as f:
    rf = pickle.load(f)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    text = data.get('article')

    if not text:
        return jsonify({"error": "No article text provided"}), 400

    # Process the text and make predictions
    # Example: Using a model for prediction
    prediction = rf.predict([text])

    return jsonify({"prediction": prediction[0]})

# Serverless handler for Vercel
def handler(request):
    return app(request)

