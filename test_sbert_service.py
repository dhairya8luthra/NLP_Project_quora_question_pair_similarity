import requests
import json

# Test SBERT service
url = 'http://localhost:5001/extract_sbert'
data = {'q1': 'How do I learn Python?', 'q2': 'What is the best way to learn Python?'}

try:
    response = requests.post(url, json=data, timeout=5)
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"Features shape: {result['shape']}")
        print(f"Actual length: {len(result['features'])}")
        print(f"First 5 features: {result['features'][:5]}")
    else:
        print(f"Error: {response.text}")
except Exception as e:
    print(f"Failed: {e}")
