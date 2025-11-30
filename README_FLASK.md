# 🔍 Quora Duplicate Question Detector - Flask Web App

A beautiful web interface for detecting duplicate questions using our trained ensemble model (85.28% accuracy).

## 📋 Prerequisites

Before running the Flask app, you need to:

### 1. Save the Models

Run the model saving cell in `complete_project.ipynb`:

- Open the notebook
- Scroll to the bottom
- Run the cell under "🔧 Save Models for Deployment"

This will create a `saved_models/` directory with all required model files.

### 2. Install Flask Dependencies

```bash
pip install -r requirements_flask.txt
```

### 3. Download spaCy Model

```bash
python -m spacy download en_core_web_sm
```

## 🚀 Running the App

1. Make sure you're in the project directory:

```bash
cd d:\CodingPlayground\NLP\Project
```

2. Run the Flask server:

```bash
python app.py
```

3. Open your browser and go to:

```
http://127.0.0.1:5000
```

## 🎯 How to Use

1. **Enter Question 1**: Type your first question in the first text box
2. **Enter Question 2**: Type your second question in the second text box
3. **Click "Check for Duplicates"**: The AI will analyze both questions
4. **View Results**:
   - See if questions are duplicates or not
   - Check the confidence score (0-100%)
   - View individual predictions from all 5 base models

## 💡 Example Questions

### Duplicate Examples:

- Question 1: "How do I learn Python?"
- Question 2: "What's the best way to learn Python?"

### Not Duplicate Examples:

- Question 1: "How do I learn Python?"
- Question 2: "What is the best Python book?"

## 🏗️ Architecture

The web app uses:

- **Backend**: Flask (Python web framework)
- **Frontend**: HTML, CSS, JavaScript (vanilla)
- **AI Models**:
  - 5 Base Models (LightGBM, XGBoost, Random Forest, Gradient Boosting, Logistic Regression)
  - Meta-learner (XGBoost Stacking Ensemble)
- **Features**: 801 engineered features per question pair

## 📁 File Structure

```
Project/
├── app.py                      # Flask application
├── templates/
│   └── index.html             # Web interface
├── saved_models/              # Trained models (created after running save cell)
│   ├── lightgbm_model.pkl
│   ├── xgboost_model.pkl
│   ├── random_forest_model.pkl
│   ├── gradient_boosting_model.pkl
│   ├── logistic_regression_model.pkl
│   ├── meta_xgboost_ensemble.pkl
│   └── sbert_model_path.txt
├── requirements_flask.txt     # Python dependencies
└── README_FLASK.md           # This file
```

## 🔧 Features Extracted

For each question pair, the system extracts:

- **771 features**: SBERT semantic embeddings
- **9 features**: Lexical similarity (word overlap, Jaccard, etc.)
- **6 features**: Fuzzy matching (handles typos, variations)
- **6 features**: Linguistic analysis (NER, POS tags)
- **3 features**: TF-IDF weighted embeddings
- **6 features**: Graph-based patterns

**Total: 801 features**

## 📊 Model Performance

- **Test Accuracy**: 85.28%
- **F1 Score**: 80.40%
- **ROC-AUC**: 93.13%

## 🛠️ Troubleshooting

### Models not found error:

Run the model saving cell in the notebook first.

### spaCy model not found:

```bash
python -m spacy download en_core_web_sm
```

### Port already in use:

Change the port in `app.py`:

```python
app.run(debug=True, host='0.0.0.0', port=5001)  # Use 5001 instead of 5000
```

## 🎨 UI Features

- **Responsive Design**: Works on desktop and mobile
- **Beautiful Gradients**: Modern purple gradient theme
- **Animated Results**: Smooth transitions and animations
- **Confidence Visualization**: Progress bar showing confidence level
- **Model Breakdown**: See individual predictions from each base model
- **Error Handling**: Clear error messages for invalid inputs

## 🔐 Security Note

This is a development server. For production deployment:

- Use a production WSGI server (gunicorn, uWSGI)
- Add input validation and sanitization
- Implement rate limiting
- Use HTTPS
- Add authentication if needed

## 📝 License

Part of the NLP Project for CS F429 - Natural Language Processing
BITS Pilani, Hyderabad Campus

---

**Enjoy detecting duplicate questions! 🚀**
