#!/usr/bin/env python3
"""
Check Advanced ML 15m UP model accuracy
"""

import joblib
import os

model_path = "models/advanced_15m_up_model.pkl"

if os.path.exists(model_path):
    print("="*60)
    print("🚀 Advanced ML 15m UP 모델 확인")
    print("="*60)

    # Load model
    model_data = joblib.load(model_path)

    # Check structure
    if isinstance(model_data, dict):
        # Model info
        accuracy = model_data.get('accuracy', model_data.get('test_accuracy', 0))
        val_accuracy = model_data.get('val_accuracy', 0)
        train_accuracy = model_data.get('train_accuracy', 0)

        print(f"테스트 정확도: {accuracy*100:.1f}%")
        if val_accuracy:
            print(f"검증 정확도: {val_accuracy*100:.1f}%")
        if train_accuracy:
            print(f"훈련 정확도: {train_accuracy*100:.1f}%")

        # Model type
        if 'model' in model_data:
            model = model_data['model']
            print(f"모델 타입: {type(model).__name__}")

        # Features
        if 'features' in model_data:
            features = model_data['features']
            print(f"특징 개수: {len(features) if isinstance(features, list) else features}")

        # Training time
        if 'training_time' in model_data:
            print(f"훈련 시간: {model_data['training_time']:.1f}초")

        # File size
        file_size = os.path.getsize(model_path) / (1024*1024)
        print(f"파일 크기: {file_size:.1f} MB")

        # Additional info
        for key in ['optimizer', 'learning_rate', 'epochs', 'batch_size']:
            if key in model_data:
                print(f"{key}: {model_data[key]}")

    else:
        print(f"모델 타입: {type(model_data).__name__}")

else:
    print(f"❌ 모델 파일이 없습니다: {model_path}")

# Check for other advanced models
print("\n📁 Advanced ML 모델 목록:")
for f in sorted(os.listdir("models")):
    if f.startswith("advanced_"):
        size = os.path.getsize(f"models/{f}") / (1024*1024)
        print(f"  - {f} ({size:.1f} MB)")