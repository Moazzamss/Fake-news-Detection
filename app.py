from flask import Flask, render_template, request, redirect, url_for
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

def majority_voting(*predictions):
    """
    This function returns the final prediction based on the majority vote.
    - If more than 2 models predict FAKE (1), classify as FAKE.
    - Otherwise, classify as REAL (0).
    """
    # Flatten the predictions into a 1D array
    predictions_flat = np.concatenate(predictions)

    # Count how many models predict 1 (FAKE)
    fake_count = np.sum(predictions_flat == 1)

    # Debugging: Print out predictions and fake count
    print(f"Flattened Predictions: {predictions_flat}, Fake Count: {fake_count}")

    # If more than 2 models predict FAKE, classify as FAKE
    if fake_count > 1:
        return 1  # FAKE
    else:
        return 0  # REAL

# Route for Model Performance page
@app.route('/model-performance')
def model_performance():
    return render_template('model_performance.html')

# Route for About page
@app.route('/about')
def about():
    return render_template('about.html')

# Home route (Fake News Detection Form)
@app.route('/')
def home():
    return render_template('index.html')

# Prediction route (Process prediction)
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get the text from the form
        text = request.form['article']
        
        if not text:
            return render_template('index.html', result="Error: No article text provided", model_predictions="")

        # Preprocess and vectorize the input text
        cleaned_text = text.lower()  # Apply text cleaning if needed
        vectorized_text = vectorizer.transform([cleaned_text])

        # Make predictions with each model
        rf_pred = rf.predict(vectorized_text)
        dt_pred = dt.predict(vectorized_text)
        lr_pred = lr.predict(vectorized_text)
        nb_pred = nb.predict(vectorized_text)

        # Debugging: Print out individual predictions
        print(f"Individual Predictions - RF: {rf_pred}, DT: {dt_pred}, LR: {lr_pred}, NB: {nb_pred}")

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

        # Debugging: Print the final result and the model predictions
        print(f"Final Prediction: {result}, Model Predictions: {model_predictions}")

        return render_template('index.html', result=result, model_predictions=model_predictions.to_html())

# News selection route (Page to pick predefined news article)
@app.route('/news-selection', methods=['GET', 'POST'])
def news_selection():
    if request.method == 'POST':
        # Get the selected article from the dropdown
        article = request.form['article']
        
        # Use the selected article and process it for prediction
        result, model_predictions = get_predictions(article)

        return render_template('index.html', result=result, model_predictions=model_predictions.to_html())

    # Show predefined news articles for selection
    return render_template('news_selection.html')

# Helper function to handle article prediction
def get_predictions(text):
    cleaned_text = text.lower()  # Clean the input text
    vectorized_text = vectorizer.transform([cleaned_text])

    # Make predictions with each model
    rf_pred = rf.predict(vectorized_text)
    dt_pred = dt.predict(vectorized_text)
    lr_pred = lr.predict(vectorized_text)
    nb_pred = nb.predict(vectorized_text)

    # Debugging: Print out individual predictions
    print(f"Individual Predictions - RF: {rf_pred}, DT: {dt_pred}, LR: {lr_pred}, NB: {nb_pred}")

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

    return result, model_predictions

if __name__ == '__main__':
    app.run(debug=True)
