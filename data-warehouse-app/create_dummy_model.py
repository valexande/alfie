#!/usr/bin/env python3
"""
Create a dummy model.pkl file for testing the AutoML consumer

This creates a simple pickle file that can be uploaded to the Data Warehouse
to test the complete Kafka orchestration flow without implementing actual AutoML.
"""

import pickle

# Create a simple dummy model object
dummy_model = {
    'type': 'classifier',
    'algorithm': 'RandomForest',
    'accuracy': 0.92,
    'features': ['feature_1', 'feature_2', 'feature_3'],
    'classes': ['class_0', 'class_1'],
    'created_for': 'testing Kafka flow',
    'note': 'This is a dummy model for testing purposes only'
}

# Save it as model.pkl
with open('model.pkl', 'wb') as f:
    pickle.dump(dummy_model, f)

print("✅ Created dummy model.pkl")
print(f"   Size: {len(pickle.dumps(dummy_model))} bytes")
print("   Location: ./model.pkl")
print("\nYou can now test the AutoML consumer!")

