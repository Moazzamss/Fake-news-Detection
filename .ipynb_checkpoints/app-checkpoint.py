from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd

# Initialize Flask app
app = Flask(__name__)

# Load the models and vectorizer
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

# Majority voting function
def majority_voting(*predictions):
    return np.sign(np.sum(predictions, axis=0))

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get the text from the form
        text = request.form['text']
        
        # Preprocess and vectorize the input text
        cleaned_text = text.lower()  # Apply text cleaning if needed
        vectorized_text = vectorizer.transform([cleaned_text])
        
        # Make predictions with each model
        rf_pred = rf.predict(vectorized_text)
        dt_pred = dt.predict(vectorized_text)
        lr_pred = lr.predict(vectorized_text)
        nb_pred = nb.predict(vectorized_text)
        
        # Majority Voting
        predictions = np.array([rf_pred, dt_pred, lr_pred, nb_pred])
        final_pred = majority_voting(*predictions)

        # Convert predictions to 'REAL' or 'FAKE'
        result = 'FAKE' if final_pred == 1 else 'REAL'

        # Prepare individual model predictions
        model_predictions = pd.DataFrame({
            'Model': ['Random Forest', 'Decision Tree', 'Logistic Regression', 'Naive Bayes'],
            'Prediction': [rf_pred[0], dt_pred[0], lr_pred[0], nb_pred[0]]
        })

        return render_template('index.html', result=result, model_predictions=model_predictions.to_html())

if __name__ == '__main__':
    app.run(debug=True)
