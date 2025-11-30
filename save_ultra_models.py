# Script to retrain meta-learner with XGBoost and save Ultra Ensemble models
import numpy as np
import joblib
import os
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

print("="*70)
print("ULTRA ENSEMBLE - Retraining with XGBoost Meta-Learner")
print("="*70)

# Create new folder for ultra models
os.makedirs('models_ultra', exist_ok=True)

# Load existing base models and data
print("\n1️⃣ Loading existing base models...")
lgb_model = joblib.load('models/lightgbm_model.pkl')
xgb_model = joblib.load('models/xgboost_model.pkl')
rf_model = joblib.load('models/random_forest_model.pkl')
gb_model = joblib.load('models/gradient_boosting_model.pkl')
lr_model = joblib.load('models/logistic_regression_model.pkl')
scaler = joblib.load('models/scaler.pkl')
tfidf = joblib.load('models/tfidf_vectorizer.pkl')

print("✓ All base models loaded")

# Load the scaled features (these should exist from training)
print("\n2️⃣ Loading feature data...")
# Note: You need to have X_train_scaled, X_val_scaled, y_train, y_val available
# This assumes they were saved or you run this after training cells

try:
    X_val_scaled = np.load('temp_X_val_scaled.npy')
    X_test_scaled = np.load('temp_X_test_scaled.npy')
    y_val = np.load('temp_y_val.npy')
    y_test = np.load('temp_y_test.npy')
    print("✓ Loaded saved feature data")
except:
    print("❌ Error: Feature data not found!")
    print("   Please run the training cells first to generate X_val_scaled, X_test_scaled, y_val, y_test")
    print("   Or add code to save these arrays after training")
    exit(1)

# Get base model predictions
print("\n3️⃣ Getting base model predictions...")
lgb_val = lgb_model.predict(X_val_scaled)
xgb_val = xgb_model.predict(X_val_scaled)
rf_val = rf_model.predict_proba(X_val_scaled)[:, 1]
gb_val = gb_model.predict_proba(X_val_scaled)[:, 1]
lr_val = lr_model.predict_proba(X_val_scaled)[:, 1]

lgb_test = lgb_model.predict(X_test_scaled)
xgb_test = xgb_model.predict(X_test_scaled)
rf_test = rf_model.predict_proba(X_test_scaled)[:, 1]
gb_test = gb_model.predict_proba(X_test_scaled)[:, 1]
lr_test = lr_model.predict_proba(X_test_scaled)[:, 1]

# Stack predictions
stacking_val = np.column_stack([lgb_val, xgb_val, rf_val, gb_val, lr_val])
stacking_test = np.column_stack([lgb_test, xgb_test, rf_test, gb_test, lr_test])

print(f"  Val stacking shape: {stacking_val.shape}")
print(f"  Test stacking shape: {stacking_test.shape}")

# Train XGBoost meta-learner
print("\n4️⃣ Training XGBoost meta-learner...")
meta_xgb = XGBClassifier(
    max_depth=3,
    learning_rate=0.1,
    n_estimators=50,
    random_state=42,
    eval_metric='logloss'
)

meta_xgb.fit(stacking_val, y_val)
print("✓ XGBoost meta-learner trained")

# Evaluate
print("\n5️⃣ Evaluating Ultra Ensemble...")
val_pred = meta_xgb.predict(stacking_val)
val_prob = meta_xgb.predict_proba(stacking_val)[:, 1]
test_pred = meta_xgb.predict(stacking_test)
test_prob = meta_xgb.predict_proba(stacking_test)[:, 1]

val_acc = accuracy_score(y_val, val_pred)
val_f1 = f1_score(y_val, val_pred)
val_auc = roc_auc_score(y_val, val_prob)

test_acc = accuracy_score(y_test, test_pred)
test_f1 = f1_score(y_test, test_pred)
test_auc = roc_auc_score(y_test, test_prob)

print("\n📊 Validation Metrics:")
print(f"  Accuracy:  {val_acc*100:.2f}%")
print(f"  F1 Score:  {val_f1*100:.2f}%")
print(f"  ROC-AUC:   {val_auc*100:.2f}%")

print("\n📊 Test Metrics:")
print(f"  Accuracy:  {test_acc*100:.2f}%")
print(f"  F1 Score:  {test_f1*100:.2f}%")
print(f"  ROC-AUC:   {test_auc*100:.2f}%")

# Save all models to models_ultra/
print("\n6️⃣ Saving models to models_ultra/...")
joblib.dump(lgb_model, 'models_ultra/lightgbm_model.pkl')
print("✓ Saved: LightGBM")

joblib.dump(xgb_model, 'models_ultra/xgboost_model.pkl')
print("✓ Saved: XGBoost")

joblib.dump(rf_model, 'models_ultra/random_forest_model.pkl')
print("✓ Saved: Random Forest")

joblib.dump(gb_model, 'models_ultra/gradient_boosting_model.pkl')
print("✓ Saved: Gradient Boosting")

joblib.dump(lr_model, 'models_ultra/logistic_regression_model.pkl')
print("✓ Saved: Logistic Regression")

joblib.dump(meta_xgb, 'models_ultra/meta_learner_xgb.pkl')
print("✓ Saved: Meta-Learner (XGBoost)")

joblib.dump(scaler, 'models_ultra/scaler.pkl')
print("✓ Saved: Scaler")

joblib.dump(tfidf, 'models_ultra/tfidf_vectorizer.pkl')
print("✓ Saved: TF-IDF Vectorizer")

print("\n" + "="*70)
print(f"✅ ULTRA ENSEMBLE MODELS SAVED TO models_ultra/")
print(f"   Feature count: 786 (6 leak + 9 lexical + 771 SBERT)")
print(f"   Meta-learner: XGBoost (not Ridge)")
print(f"   Test Accuracy: {test_acc*100:.2f}%")
print("="*70)
