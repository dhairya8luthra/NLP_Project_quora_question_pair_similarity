import sys
import traceback

print("Testing sentence_transformers import...")
sys.stdout.flush()

try:
    from sentence_transformers import SentenceTransformer
    print("✅ Import successful!")
    print(f"Version: {SentenceTransformer.__module__}")
except Exception as e:
    print(f"❌ Import failed: {e}")
    traceback.print_exc()
    sys.stdout.flush()
