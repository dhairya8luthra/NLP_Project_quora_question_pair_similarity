from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
from sentence_transformers import SentenceTransformer
import networkx as nx
from collections import defaultdict
import warnings
import sys
import traceback
warnings.filterwarnings('ignore')

app = Flask(__name__)

# Load models
print("Loading models...", flush=True)
try:
    # Load base models
    lgb_model = joblib.load('models/lightgbm_model.pkl')
    xgb_model = joblib.load('models/xgboost_model.pkl')
    rf_model = joblib.load('models/random_forest_model.pkl')
    gb_model = joblib.load('models/gradient_boosting_model.pkl')
    lr_model = joblib.load('models/logistic_regression_model.pkl')
    
    # Load meta-learner (ensemble)
    meta_learner = joblib.load('models/meta_learner.pkl')
    
    # Load scaler and vectorizer
    scaler = joblib.load('models/scaler.pkl')
    tfidf_vectorizer = joblib.load('models/tfidf_vectorizer.pkl')
    
    # Load SBERT model from local directory
    print("Loading SBERT model from local directory...")
    sbert_model = SentenceTransformer('models/sbert_local')
    print("✅ SBERT model loaded successfully!")
    
    # Initialize question graph for leak features
    q_graph = defaultdict(list)
    
    print("✅ All models loaded successfully!", flush=True)
except Exception as e:
    print(f"❌ Error loading models: {e}", flush=True)
    traceback.print_exc(file=sys.stdout)
    sys.stdout.flush()
    sys.exit(1)

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
        features.append(common_neighbors)  # common_neighbors
        
        # PageRank
        try:
            pagerank = nx.pagerank(G)
            features.append(abs(pagerank.get(q1, 0) - pagerank.get(q2, 0)))  # pagerank_diff
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
    """Extract SBERT embedding features"""
    # Encode questions
    q1_emb = sbert_model.encode(q1)
    q2_emb = sbert_model.encode(q2)
    
    # Concatenate embeddings: [q1_emb, q2_emb, abs_diff]
    abs_diff = np.abs(q1_emb - q2_emb)
    
    # Similarity metrics
    cosine_sim = np.dot(q1_emb, q2_emb) / (np.linalg.norm(q1_emb) * np.linalg.norm(q2_emb))
    euclidean_dist = np.linalg.norm(q1_emb - q2_emb)
    dot_product = np.dot(q1_emb, q2_emb)
    
    # Combine all (384 + 384 + 384 + 3 = 771 features)
    features = np.concatenate([
        q1_emb,
        q2_emb,
        abs_diff,
        [cosine_sim, euclidean_dist, dot_product]
    ])
    
    return features

def extract_all_features(q1, q2):
    """Extract all 786 features"""
    # Extract all feature types
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
    try:
        data = request.get_json()
        q1 = data.get('question1', '')
        q2 = data.get('question2', '')
        
        if not q1 or not q2:
            return jsonify({'error': 'Both questions are required'}), 400
        
        # Extract features
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
            'message': 'These questions are duplicates!' if ensemble_pred else 'These questions are NOT duplicates.'
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚀 Starting Quora Duplicate Question Detection Server")
    print("="*60)
    print("📊 Models loaded: 5 base models + 1 meta-learner")
    print("🎯 Feature extraction: 786 features (leak + lexical + SBERT)")
    print("🌐 Server running at: http://localhost:5000")
    print("="*60 + "\n")
    app.run(debug=True, port=5000)
