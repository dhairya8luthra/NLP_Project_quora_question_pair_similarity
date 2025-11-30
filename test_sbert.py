import os
import sys
from sentence_transformers import SentenceTransformer

print("Testing SBERT loading from cache...")

cache_path = os.path.join(os.path.expanduser('~'), '.cache', 'huggingface', 'hub', 'models--sentence-transformers--all-MiniLM-L6-v2')
print(f"Cache path: {cache_path}")
print(f"Cache exists: {os.path.exists(cache_path)}")

if os.path.exists(cache_path):
    snapshots_dir = os.path.join(cache_path, 'snapshots')
    if os.path.exists(snapshots_dir):
        snapshot = os.listdir(snapshots_dir)[0]
        model_path = os.path.join(snapshots_dir, snapshot)
        print(f"Loading from: {model_path}")
        try:
            model = SentenceTransformer(model_path)
            print("✅ Model loaded successfully!")
            
            # Test encoding
            test_text = "This is a test"
            embedding = model.encode(test_text)
            print(f"✅ Test encoding successful! Shape: {embedding.shape}")
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
