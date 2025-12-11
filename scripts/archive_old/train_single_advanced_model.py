#!/usr/bin/env python3
"""
단일 Advanced ML Model 훈련 (15m UP 모델)
Transformer 포함 전체 기법 사용
"""

from train_advanced_ml_models import AdvancedMLTrainer
from datetime import datetime

def main():
    print("="*60)
    print("🚀 Advanced ML 모델 훈련 (15m UP)")
    print("📅 " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("🤖 Transformer 포함")
    print("="*60)

    trainer = AdvancedMLTrainer()

    # 15분 상승 모델만 훈련
    result = trainer.train_specialist_model('15m', 'up')

    print("\n" + "="*60)
    print("✅ 훈련 완료!")
    print("="*60)

    if result:
        print(f"최고 모델: {result['best_model']}")
        print(f"최고 정확도: {result['best_accuracy']*100:.1f}%")
        print(f"앙상블 정확도: {result['ensemble_accuracy']*100:.1f}%")

if __name__ == "__main__":
    main()