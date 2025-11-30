import sys
import traceback

try:
    print("Starting Flask app import...", flush=True)
    import app
    print("App imported successfully!", flush=True)
except Exception as e:
    print(f"FATAL ERROR: {e}", flush=True)
    traceback.print_exc(file=sys.stdout)
    sys.stdout.flush()
    sys.exit(1)
