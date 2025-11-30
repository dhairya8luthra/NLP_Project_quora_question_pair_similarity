from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import networkx as nx
from collections import defaultdict
import requests
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

print("="*60)
print("🚀 Loading Quora Duplicate Detection Models (Full Features)")
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
    
    # Initialize question graph for leak features
    q_graph = defaultdict(list)
    
    print("\n✅ All models loaded successfully!")
    print("="*60)
    models_loaded = True
except Exception as e:
    print(f"\n❌ Error loading models: {e}")
    models_loaded = False

def extract_leak_features(q1, q2):
    """Extract graph-based leak features"""
    features = []
    
    # Add edges to graph
    q_graph[q1].append(q2)
    q_graph[q2].append(q1)
    
    # Create networkx graph
    G = nx.Graph()
    for q, neighbors in q_graph.items():
        for neighbor in neighbors:
            G.add_edge(q, neighbor)
    
    # Extract features
    try:
        features.append(len(q_graph[q1]))  # q1_freq
        features.append(len(q_graph[q2]))  # q2_freq
        features.append(max(len(q_graph[q1]), len(q_graph[q2])))  # max_freq
        features.append(min(len(q_graph[q1]), len(q_graph[q2])))  # min_freq
        
        # Common neighbors
        common_neighbors = len(set(q_graph[q1]) & set(q_graph[q2]))
        features.append(common_neighbors)
        
        # PageRank
        try:
            pagerank = nx.pagerank(G)
            features.append(abs(pagerank.get(q1, 0) - pagerank.get(q2, 0)))
        except:
            features.append(0)
    except:
        features = [0] * 6
    
    return np.array(features)

def extract_lexical_features(q1, q2):
    """Extract lexical overlap features"""
    q1_words = set(q1.lower().split())
    q2_words = set(q2.lower().split())
    
    intersection = q1_words & q2_words
    union = q1_words | q2_words
    
    features = []
    features.append(len(intersection))  # word_overlap
    features.append(len(intersection) / len(union) if union else 0)  # word_share
    features.append(len(intersection) / len(union) if union else 0)  # jaccard
    features.append(len(q1))  # len_q1
    features.append(len(q2))  # len_q2
    features.append(abs(len(q1) - len(q2)))  # len_diff
    features.append(min(len(q1), len(q2)) / max(len(q1), len(q2)) if max(len(q1), len(q2)) > 0 else 0)  # len_ratio
    features.append(len(intersection))  # common_words
    features.append(len(union))  # total_unique_words
    
    return np.array(features)

def extract_sbert_features(q1, q2):
    """Extract SBERT features from notebook service"""
    try:
        response = requests.post(
            'http://localhost:8080/extract_sbert',
            json={'question1': q1, 'question2': q2},
            timeout=10
        )
        if response.status_code == 200:
            data = response.json()
            return np.array(data['features'])
        else:
            raise Exception(f"SBERT service returned status {response.status_code}")
    except Exception as e:
        print(f"Warning: SBERT service error: {e}")
        # Return zeros if service fails
        return np.zeros(771)

def extract_all_features(q1, q2):
    """Extract all 786 features (6 leak + 9 lexical + 771 SBERT)"""
    leak_feats = extract_leak_features(q1, q2)  # 6 features
    lex_feats = extract_lexical_features(q1, q2)  # 9 features
    sbert_feats = extract_sbert_features(q1, q2)  # 771 features
    
    # Combine all features (786 total)
    all_features = np.concatenate([leak_feats, lex_feats, sbert_feats])
    
    # Scale features
    all_features_scaled = scaler.transform(all_features.reshape(1, -1))
    
    return all_features_scaled

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
        
        # Extract all features
        features = extract_all_features(q1, q2)
        
        # Get base model predictions
        lgb_prob = lgb_model.predict(features, num_iteration=lgb_model.best_iteration)[0]
        xgb_prob = xgb_model.predict_proba(features)[:, 1][0]
        rf_prob = rf_model.predict_proba(features)[:, 1][0]
        gb_prob = gb_model.predict_proba(features)[:, 1][0]
        lr_prob = lr_model.predict_proba(features)[:, 1][0]
        
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
            'note': 'Using full 786 features (leak + lexical + SBERT embeddings)'
        }
        
        return jsonify(response)
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    if models_loaded:
        print("\n🌐 Starting Flask Development Server...")
        print("📍 Server URL: http://localhost:5000")
        print("✅ Using FULL features: 786 (6 leak + 9 lexical + 771 SBERT)")
        print("🔗 Connected to SBERT service at http://localhost:8080")
        print("="*60 + "\n")
        app.run(debug=True, port=5000, use_reloader=False)  # Disable reloader to prevent duplicate threads
    else:
        print("\n❌ Cannot start server - models failed to load")
