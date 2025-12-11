#!/usr/bin/env python3
"""
고정확도 모델 테스트 (작은 데이터셋)
"""

from train_high_accuracy_models import HighAccuracyTrainer
import time

def test_small_dataset():
    trainer = HighAccuracyTrainer()

    print("=" * 60)
    print("🧪 고정확도 모델 테스트 (작은 데이터셋)")
    print("=" * 60)

    start_time = time.time()

    # 15분봉만 테스트 (5000개 데이터)
    model_info = trainer.train_ensemble_model('15m', data_limit=5000)

    elapsed = time.time() - start_time

    print(f"\n⏱️ 소요 시간: {elapsed:.1f}초")

    if model_info:
        print("\n✅ 테스트 성공!")
        print(f"  정확도: {model_info['ensemble_accuracy']*100:.1f}%")
        print(f"  최고 모델: {model_info['best_single_model']}")

if __name__ == "__main__":
    test_small_dataset()