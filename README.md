Technologies Used
Backend

Flask 3.0.0: Web framework
Flask-CORS 4.0.0: Cross-origin resource sharing
scikit-learn 1.5.0: Machine learning models
NumPy 1.26.4: Numerical computing
Pandas 2.2.0: Data manipulation
Gunicorn 21.2.0: WSGI HTTP server

Frontend

HTML5/CSS3: Structure and styling
JavaScript (ES6+): Interactive functionality
Fetch API: HTTP requests to backend

Deployment

Backend: Koyeb (Free tier)
Frontend: Vercel (Free tier)

Installation
Prerequisites

Python 3.10 or higher
pip (Python package manager)
Git

How It Works

Text Preprocessing: Input text is vectorized using TF-IDF
Model Predictions: 4 models independently classify the text
Ensemble Voting:

If 2+ models predict FAKE → Final = FAKE
Otherwise → Final = REAL


Confidence Calculation: Average confidence of models that agree with final prediction

🎨 UI Features

Dark Theme: Modern, eye-friendly design
Responsive Layout: Works on desktop, tablet, and mobile
Real-time Analysis: Instant predictions with loading states
Detailed Results: Shows individual model predictions and confidences
Sample News: Pre-loaded examples to test

🙏 Acknowledgments

Dataset: Kaggle Fake News Dataset
Scikit-learn for ML algorithms
Flask for backend framework
Vercel & Koyeb for free hosting

🔮 Future Enhancements

 Add more ML models (SVM, Neural Networks)
 Real-time news scraping
 User authentication
 Save prediction history
 Mobile app version
 Multi-language support
 Explainable AI features