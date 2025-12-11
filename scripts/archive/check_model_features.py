#!/usr/bin/env python3
"""
Check Deep Ensemble model features
"""

import joblib

# Load model
model_data = joblib.load("models/deep_ensemble_15m_up_model.pkl")

print("Deep Ensemble 15m UP Model Information:")
print("="*60)
print(f"Accuracy: {model_data.get('ensemble_accuracy', 0)*100:.1f}%")
print(f"AUC: {model_data.get('ensemble_auc', 0):.3f}")
print(f"Number of models: {len(model_data.get('models', {}))}")
print(f"\nFeatures required: {len(model_data.get('features', []))}")
print("\nFeature names:")
for i, f in enumerate(model_data.get('features', [])[:20], 1):
    print(f"  {i:2d}. {f}")
if len(model_data.get('features', [])) > 20:
    print(f"  ... and {len(model_data.get('features', [])) - 20} more features")

print("\nModels included:")
for name, info in model_data.get('models', {}).items():
    acc = info.get('accuracy', 0) * 100
    print(f"  - {name}: {acc:.1f}% accuracy")

print("\nWeights:")
for name, weight in model_data.get('weights', {}).items():
    print(f"  - {name}: {weight:.2f}")