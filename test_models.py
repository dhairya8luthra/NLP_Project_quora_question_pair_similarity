import joblib
import numpy as np
from sentence_transformers import SentenceTransformer

print("Testing model loading...")

try:
    print("1. Loading LightGBM...")
    lgb_model = joblib.load('models/lightgbm_model.pkl')
    print("✅ LightGBM loaded")
    
    print("2. Loading XGBoost...")
    xgb_model = joblib.load('models/xgboost_model.pkl')
    print("✅ XGBoost loaded")
    
    print("3. Loading Random Forest...")
    rf_model = joblib.load('models/random_forest_model.pkl')
    print("✅ Random Forest loaded")
    
    print("4. Loading Gradient Boosting...")
    gb_model = joblib.load('models/gradient_boosting_model.pkl')
    print("✅ Gradient Boosting loaded")
    
    print("5. Loading Logistic Regression...")
    lr_model = joblib.load('models/logistic_regression_model.pkl')
    print("✅ Logistic Regression loaded")
    
    print("6. Loading Meta-Learner...")
    meta_learner = joblib.load('models/meta_learner.pkl')
    print("✅ Meta-Learner loaded")
    
    print("7. Loading Scaler...")
    scaler = joblib.load('models/scaler.pkl')
    print("✅ Scaler loaded")
    
    print("8. Loading TF-IDF Vectorizer...")
    tfidf_vectorizer = joblib.load('models/tfidf_vectorizer.pkl')
    print("✅ TF-IDF Vectorizer loaded")
    
    print("9. Loading SBERT model...")
    sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
    print("✅ SBERT model loaded")
    
    print("\n✅ All models loaded successfully!")
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
