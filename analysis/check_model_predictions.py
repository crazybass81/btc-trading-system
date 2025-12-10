#!/usr/bin/env python3
"""
모델의 실제 예측 분포 확인 - 백테스트 데이터 사용
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
import ccxt
from loguru import logger

# 모델과 스케일러 로드
def load_models():
    models = {}
    scalers = {}

    model_files = {
        '15m': ('models/main_15m_model.pkl', 'models/main_15m_scaler.pkl'),
        '30m': ('models/main_30m_model.pkl', 'models/main_30m_scaler.pkl'),
        '4h': ('models/trend_4h_model.pkl', 'models/trend_4h_scaler.pkl'),
        '1d': ('models/trend_1d_model.pkl', 'models/trend_1d_scaler.pkl')
    }

    for timeframe, (model_file, scaler_file) in model_files.items():
        if os.path.exists(model_file) and os.path.exists(scaler_file):
            models[timeframe] = joblib.load(model_file)
            scalers[timeframe] = joblib.load(scaler_file)
            print(f"✅ {timeframe} 모델 로드 완료")
        else:
            print(f"❌ {timeframe} 모델 파일 없음")

    return models, scalers

# 데이터 가져오기
def get_historical_data(timeframe='15m', limit=1000):
    exchange = ccxt.binance()

    # 타임프레임 매핑
    tf_map = {
        '15m': '15m',
        '30m': '30m',
        '4h': '4h',
        '1d': '1d'
    }

    try:
        ohlcv = exchange.fetch_ohlcv(
            'BTC/USDT',
            timeframe=tf_map[timeframe],
            limit=limit
        )

        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)

        return df
    except Exception as e:
        print(f"데이터 가져오기 실패: {e}")
        return None

# 특징 생성 (15분 모델용)
def prepare_basic_features(df):
    features = pd.DataFrame(index=df.index)

    # 가격 변화율
    for period in [1, 3, 5, 10]:
        features[f'return_{period}'] = df['close'].pct_change(period) * 100

    # RSI
    for period in [7, 14, 21]:
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss.replace(0, 1)
        features[f'rsi_{period}'] = 100 - (100 / (1 + rs))

    # MACD
    exp1 = df['close'].ewm(span=12, adjust=False).mean()
    exp2 = df['close'].ewm(span=26, adjust=False).mean()
    features['macd'] = exp1 - exp2
    features['macd_signal'] = features['macd'].ewm(span=9, adjust=False).mean()

    # 볼린저 밴드
    for period in [10, 20]:
        sma = df['close'].rolling(window=period).mean()
        std = df['close'].rolling(window=period).std()
        features[f'bb_width_{period}'] = (std * 2) / sma * 100
        features[f'bb_position_{period}'] = (df['close'] - sma) / (std * 2)

    # 볼륨
    features['volume_ratio'] = df['volume'] / df['volume'].rolling(window=20).mean()
    features['volume_change'] = df['volume'].pct_change() * 100

    # High-Low 비율
    features['high_low_ratio'] = (df['high'] - df['low']) / df['close'] * 100

    return features.fillna(0)

# 예측 테스트
def test_predictions(models, scalers):
    results = {}

    for timeframe in ['15m', '30m', '4h', '1d']:
        if timeframe not in models:
            continue

        print(f"\n{'='*60}")
        print(f"📊 {timeframe} 모델 테스트")
        print(f"{'='*60}")

        # 데이터 가져오기
        df = get_historical_data(timeframe, limit=500)
        if df is None:
            continue

        # 특징 생성 (간단화)
        features = prepare_basic_features(df)

        # 최근 100개 데이터로 테스트
        test_features = features.iloc[-100:]

        # 예측 분포 확인
        predictions = []
        probabilities = []

        for i in range(len(test_features)):
            try:
                # 스케일링
                X = test_features.iloc[i:i+1]

                # 15m 모델은 16개 특징 선택
                if timeframe == '15m':
                    feature_cols = ['return_1', 'return_3', 'return_5', 'return_10',
                                  'rsi_7', 'rsi_14', 'rsi_21', 'macd', 'macd_signal',
                                  'bb_width_10', 'bb_width_20', 'bb_position_10',
                                  'bb_position_20', 'volume_ratio', 'volume_change',
                                  'high_low_ratio']
                    X = X[feature_cols]

                # 30m 모델은 30개 특징 필요 - 실제 모델과 맞춰야 함
                elif timeframe == '30m':
                    # 30분 모델은 특별 처리 필요
                    continue

                # 스케일링
                X_scaled = scalers[timeframe].transform(X)

                # 예측
                pred = models[timeframe].predict(X_scaled)[0]
                pred_proba = models[timeframe].predict_proba(X_scaled)[0]

                predictions.append(pred)
                probabilities.append(max(pred_proba))

            except Exception as e:
                continue

        if predictions:
            # 결과 분석
            unique, counts = np.unique(predictions, return_counts=True)
            total = len(predictions)

            print(f"총 {total}개 예측:")

            # 신호 매핑
            signal_map = {0: 'SHORT', 1: 'NEUTRAL', 2: 'LONG'}

            for val, count in zip(unique, counts):
                signal = signal_map.get(val, f'Unknown({val})')
                percentage = (count / total) * 100
                print(f"  {signal:8s}: {count:3d}회 ({percentage:5.1f}%)")

            # 평균 신뢰도
            if probabilities:
                avg_prob = np.mean(probabilities) * 100
                print(f"\n평균 신뢰도: {avg_prob:.1f}%")

            # 최근 10개 예측
            print(f"\n최근 10개 예측:")
            for i, (pred, prob) in enumerate(list(zip(predictions[-10:], probabilities[-10:])), 1):
                signal = signal_map.get(pred, f'Unknown({pred})')
                print(f"  {i:2d}. {signal:8s} (신뢰도: {prob*100:.1f}%)")

            results[timeframe] = {
                'predictions': predictions,
                'distribution': dict(zip(unique, counts))
            }

    return results

# 문제 진단
def diagnose_neutral_bias():
    print("\n" + "="*60)
    print("🔍 NEUTRAL 편향 문제 진단")
    print("="*60)

    # 15분 모델 상세 분석
    model_15m = joblib.load('models/main_15m_model.pkl')
    scaler_15m = joblib.load('models/main_15m_scaler.pkl')

    # 임계값 확인
    print("\n1. 모델 클래스 분포:")
    if hasattr(model_15m, 'classes_'):
        print(f"   클래스: {model_15m.classes_}")

    if hasattr(model_15m, 'class_weight_'):
        print(f"   클래스 가중치: {model_15m.class_weight_}")

    # 특징 중요도
    if hasattr(model_15m, 'feature_importances_'):
        print(f"\n2. 특징 중요도 상위 5개:")
        feature_names = ['return_1', 'return_3', 'return_5', 'return_10',
                        'rsi_7', 'rsi_14', 'rsi_21', 'macd', 'macd_signal',
                        'bb_width_10', 'bb_width_20', 'bb_position_10',
                        'bb_position_20', 'volume_ratio', 'volume_change',
                        'high_low_ratio']

        importances = model_15m.feature_importances_
        indices = np.argsort(importances)[::-1][:5]

        for i in indices:
            print(f"   {feature_names[i]}: {importances[i]:.4f}")

    print("\n3. 가능한 원인:")
    print("   - 훈련 데이터의 클래스 불균형")
    print("   - 임계값 설정 문제")
    print("   - 특징 스케일링 문제")
    print("   - 과적합/과소적합")

    print("\n4. 해결 방법:")
    print("   - 클래스 가중치 조정")
    print("   - 임계값 최적화")
    print("   - 더 다양한 시장 상황 데이터로 재훈련")
    print("   - 앙상블 방법 개선")

if __name__ == "__main__":
    # 모델 로드
    models, scalers = load_models()

    # 예측 테스트
    results = test_predictions(models, scalers)

    # 문제 진단
    diagnose_neutral_bias()

    print("\n" + "="*60)
    print("✅ 분석 완료!")
    print("="*60)