"""Quick test script for the LRI prediction backend."""

import json
from inference import LRIPredictor

# Initialize predictor
pred = LRIPredictor(models_dir='models')

# Load test data from JSON file
with open('test_request.json', 'r') as f:
    patient_data = json.load(f)

print('='*60)
print('Testing LRI Prediction')
print('='*60)
print(f"Input cylinder: {patient_data['manifest_cylinder']} @ {patient_data['manifest_axis']}°")
result = pred.predict(patient_data)
print(f"Result: {result.to_dict()}")

print()
print('='*60)
print('Test completed successfully!')
print('='*60)
