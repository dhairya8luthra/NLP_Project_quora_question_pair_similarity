from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

print("="*60)
print("🚀 Loading Quora Duplicate Detection Models...")
print("="*60)

# Load models
try:
    print("Loading base models...")
    lgb_model = joblib.load('models/lightgbm_model.pkl')
    print("✓ LightGBM loaded")
    
    xgb_model = joblib.load('models/xgboost_model.pkl')
    print("✓ XGBoost loaded")
    
    rf_model = joblib.load('models/random_forest_model.pkl')
    print("✓ Random Forest loaded")
    
    gb_model = joblib.load('models/gradient_boosting_model.pkl')
    print("✓ Gradient Boosting loaded")
    
    lr_model = joblib.load('models/logistic_regression_model.pkl')
    print("✓ Logistic Regression loaded")
    
    # Load meta-learner
    meta_learner = joblib.load('models/meta_learner.pkl')
    print("✓ Meta-Learner (Ridge) loaded")
    
    # Load scaler
    scaler = joblib.load('models/scaler.pkl')
    print("✓ StandardScaler loaded")
    
    print("\n✅ All models loaded successfully!")
    print("="*60)
    models_loaded = True
except Exception as e:
    print(f"\n❌ Error loading models: {e}")
    models_loaded = False

# Simplified feature extraction (without SBERT for now)
def extract_simple_features(q1, q2):
    """Extract basic lexical features only"""
    q1_words = set(q1.lower().split())
    q2_words = set(q2.lower().split())
    
    intersection = q1_words & q2_words
    union = q1_words | q2_words
    
    features = []
    # Leak features (simplified - all zeros for new questions)
    features.extend([0, 0, 0, 0, 0, 0])  # 6 leak features
    
    # Lexical features
    features.append(len(intersection))  # word_overlap
    features.append(len(intersection) / len(union) if union else 0)  # word_share
    features.append(len(intersection) / len(union) if union else 0)  # jaccard
    features.append(len(q1))  # len_q1
    features.append(len(q2))  # len_q2
    features.append(abs(len(q1) - len(q2)))  # len_diff
    features.append(min(len(q1), len(q2)) / max(len(q1), len(q2)) if max(len(q1), len(q2)) > 0 else 0)  # len_ratio
    features.append(len(intersection))  # common_words
    features.append(len(union))  # total_unique_words
    
    # Pad with zeros for missing SBERT features (771 features)
    features.extend([0] * 771)
    
    return np.array(features).reshape(1, -1)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if not models_loaded:
        return jsonify({'error': 'Models not loaded. Please check server logs.'}), 500
    
    try:
        data = request.get_json()
        q1 = data.get('question1', '')
        q2 = data.get('question2', '')
        
        if not q1 or not q2:
            return jsonify({'error': 'Both questions are required'}), 400
        
        # Extract simple features
        features = extract_simple_features(q1, q2)
        
        # Scale features
        features_scaled = scaler.transform(features)
        
        # Get base model predictions
        lgb_prob = lgb_model.predict(features_scaled, num_iteration=lgb_model.best_iteration)[0]
        xgb_prob = xgb_model.predict_proba(features_scaled)[:, 1][0]
        rf_prob = rf_model.predict_proba(features_scaled)[:, 1][0]
        gb_prob = gb_model.predict_proba(features_scaled)[:, 1][0]
        lr_prob = lr_model.predict_proba(features_scaled)[:, 1][0]
        
        # Stack predictions for meta-learner
        stacked_features = np.array([[lgb_prob, xgb_prob, rf_prob, gb_prob, lr_prob]])
        
        # Get ensemble prediction
        ensemble_prob = meta_learner.predict(stacked_features)[0]
        ensemble_pred = 1 if ensemble_prob > 0.5 else 0
        
        response = {
            'duplicate': bool(ensemble_pred),
            'confidence': float(ensemble_prob),
            'base_predictions': {
                'lightgbm': float(lgb_prob),
                'xgboost': float(xgb_prob),
                'random_forest': float(rf_prob),
                'gradient_boosting': float(gb_prob),
                'logistic_regression': float(lr_prob)
            },
            'message': 'These questions are duplicates!' if ensemble_pred else 'These questions are NOT duplicates.',
            'note': 'Using simplified features (lexical only) - SBERT embeddings not available'
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    if models_loaded:
        print("\n🌐 Starting Flask Development Server...")
        print("📍 Server URL: http://localhost:5000")
        print("⚠️  Note: Using simplified features (lexical only)")
        print("💡 Full SBERT features available in notebook environment")
        print("="*60 + "\n")
        app.run(debug=True, port=5000)
    else:
        print("\n❌ Cannot start server - models failed to load")
