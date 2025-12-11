#!/usr/bin/env python3
"""
Test Deep Ensemble 15m UP Model (62.8% accuracy)
실시간 예측 테스트
"""

import ccxt
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def test_deep_ensemble():
    print("="*60)
    print("🎯 Deep Ensemble 15m UP 모델 테스트 (62.8% 정확도)")
    print("📅 " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("="*60)

    # Load model
    try:
        model_data = joblib.load("models/deep_ensemble_15m_up_model.pkl")
        print("✅ 모델 로드 성공")

        # Model info
        accuracy = model_data.get('ensemble_accuracy', 0) * 100
        auc = model_data.get('ensemble_auc', 0)
        n_models = len(model_data.get('models', {}))
        print(f"  정확도: {accuracy:.1f}%")
        print(f"  AUC: {auc:.3f}")
        print(f"  앙상블 모델 수: {n_models}")

    except Exception as e:
        print(f"❌ 모델 로드 실패: {e}")
        return

    # Get real-time data
    print("\n📊 실시간 데이터 수집...")
    exchange = ccxt.binance()
    ohlcv = exchange.fetch_ohlcv('BTC/USDT', '15m', limit=200)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    print(f"  ✅ {len(df)}개 캔들 수집")

    # Create features
    print("  📐 특징 생성 중...")
    features = create_15m_up_features(df)

    # Get scalers and models
    scalers = model_data.get('scalers', {})
    models = model_data.get('models', {})
    feature_names = model_data.get('features', [])
    weights = model_data.get('weights', {})

    # Select features
    X = features[feature_names]
    valid_idx = ~X.isna().any(axis=1)
    X_valid = X[valid_idx]

    if len(X_valid) < 10:
        print("  ⚠️ 데이터 부족")
        return

    # Make predictions
    print("\n🔮 예측 수행...")
    all_probabilities = []

    for model_name, model_info in models.items():
        if 'model' in model_info:
            model = model_info['model']
            scaler = scalers.get(model_name)
            if scaler:
                X_scaled = scaler.transform(X_valid)
                try:
                    if hasattr(model, 'predict_proba'):
                        prob = model.predict_proba(X_scaled)[:, 1]
                    else:
                        prob = model.predict(X_scaled)
                    weight = weights.get(model_name, 1.0)
                    all_probabilities.append(prob * weight)
                    print(f"  {model_name}: 예측 완료 (가중치: {weight:.2f})")
                except Exception as e:
                    print(f"  {model_name}: 예측 실패 - {e}")

    # Weighted ensemble
    if all_probabilities:
        ensemble_prob = np.mean(all_probabilities, axis=0)
        ensemble_pred = (ensemble_prob > 0.5).astype(int)

        # Latest predictions
        latest_pred = ensemble_pred[-10:]
        latest_prob = ensemble_prob[-10:]
        latest_times = X_valid.index[-10:]

        print(f"\n📈 최근 10개 UP 신호 예측:")
        print("-"*40)
        up_count = 0
        for time, pred, prob in zip(latest_times, latest_pred, latest_prob):
            if pred == 1:  # UP prediction
                confidence = prob
                print(f"  {time.strftime('%H:%M')}: 📈 UP 신호 (신뢰도: {confidence:.1%})")
                up_count += 1
            else:
                print(f"  {time.strftime('%H:%M')}: - (UP 확률: {prob:.1%})")

        print(f"\n📊 통계:")
        print(f"  UP 신호 발생: {up_count}/10 ({up_count*10}%)")
        print(f"  평균 UP 확률: {np.mean(ensemble_prob[-10:]):.1%}")

        # Strong signals
        strong_signals = ensemble_prob[-50:] > 0.6
        print(f"  강한 신호 (>60%): {np.sum(strong_signals)}/50개")

        # Current prediction
        current_prob = ensemble_prob[-1]
        current_pred = ensemble_pred[-1]
        print(f"\n🎯 현재 예측:")
        if current_pred == 1:
            print(f"  📈 UP 신호 발생! (신뢰도: {current_prob:.1%})")
        else:
            print(f"  대기 (UP 확률: {current_prob:.1%})")

    else:
        print("❌ 예측 실패")

def create_15m_up_features(df):
    """Create 15m UP-specific features"""
    features = pd.DataFrame(index=df.index)

    # Returns
    for period in [1, 3, 5, 8, 13, 21, 34, 55, 89]:
        features[f'return_{period}'] = df['close'].pct_change(period)
        features[f'return_{period}_abs'] = features[f'return_{period}'].abs()

    # Return statistics
    for window in [5, 10, 20]:
        ret = df['close'].pct_change()
        features[f'return_mean_{window}'] = ret.rolling(window).mean()
        features[f'return_std_{window}'] = ret.rolling(window).std()
        features[f'return_skew_{window}'] = ret.rolling(window).skew()
        features[f'return_kurt_{window}'] = ret.rolling(window).kurt()

    # RSI variations
    for period in [7, 14, 21, 28]:
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / (loss + 1e-10)
        features[f'rsi_{period}'] = 100 - (100 / (1 + rs))
        features[f'rsi_{period}_sma'] = features[f'rsi_{period}'].rolling(5).mean()
        features[f'rsi_{period}_std'] = features[f'rsi_{period}'].rolling(5).std()

    # Moving averages
    for short, long in [(5, 10), (10, 20), (20, 50), (50, 100), (100, 200)]:
        if len(df) > long:
            sma_short = df['close'].rolling(short).mean()
            sma_long = df['close'].rolling(long).mean()
            features[f'sma_{short}_{long}_ratio'] = (sma_short - sma_long) / (sma_long + 1e-10)

            ema_short = df['close'].ewm(span=short, adjust=False).mean()
            ema_long = df['close'].ewm(span=long, adjust=False).mean()
            features[f'ema_{short}_{long}_ratio'] = (ema_short - ema_long) / (ema_long + 1e-10)

    # MACD variations
    for fast, slow, signal in [(12, 26, 9), (5, 35, 5), (10, 20, 5)]:
        exp1 = df['close'].ewm(span=fast, adjust=False).mean()
        exp2 = df['close'].ewm(span=slow, adjust=False).mean()
        macd = exp1 - exp2
        macd_signal = macd.ewm(span=signal, adjust=False).mean()
        features[f'macd_{fast}_{slow}'] = macd / (df['close'] + 1e-10)
        features[f'macd_signal_{fast}_{slow}'] = macd_signal / (df['close'] + 1e-10)
        features[f'macd_hist_{fast}_{slow}'] = (macd - macd_signal) / (df['close'] + 1e-10)

    # Bollinger Bands variations
    for period in [10, 20, 30]:
        sma = df['close'].rolling(period).mean()
        std = df['close'].rolling(period).std()
        features[f'bb_upper_{period}'] = (df['close'] - (sma + 2*std)) / (df['close'] + 1e-10)
        features[f'bb_lower_{period}'] = ((sma - 2*std) - df['close']) / (df['close'] + 1e-10)
        features[f'bb_width_{period}'] = (4*std) / (sma + 1e-10)
        features[f'bb_position_{period}'] = (df['close'] - (sma - 2*std)) / (4*std + 1e-10)

    # Volume features
    features['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
    features['volume_trend'] = df['volume'].rolling(5).mean() / df['volume'].rolling(20).mean()
    features['volume_rsi'] = 100 - (100 / (1 + df['volume'].rolling(14).mean() / df['volume'].rolling(14).std()))

    # Price position
    for period in [20, 50, 100]:
        if len(df) > period:
            highest = df['high'].rolling(period).max()
            lowest = df['low'].rolling(period).min()
            features[f'price_position_{period}'] = (df['close'] - lowest) / (highest - lowest + 1e-10)
            features[f'dist_from_high_{period}'] = (highest - df['close']) / (df['close'] + 1e-10)
            features[f'dist_from_low_{period}'] = (df['close'] - lowest) / (df['close'] + 1e-10)

    # Volatility
    for period in [5, 10, 20, 30]:
        features[f'volatility_{period}'] = df['close'].pct_change().rolling(period).std()
        features[f'atr_{period}'] = (df['high'] - df['low']).rolling(period).mean() / df['close']

    # Pattern features
    features['higher_high'] = ((df['high'] > df['high'].shift(1)) &
                              (df['low'] > df['low'].shift(1))).astype(int)
    features['lower_low'] = ((df['high'] < df['high'].shift(1)) &
                            (df['low'] < df['low'].shift(1))).astype(int)

    # Time features
    features['hour'] = df.index.hour
    features['minute'] = df.index.minute
    features['day_of_week'] = df.index.dayofweek

    # Asia session (0-8 UTC)
    features['asia_session'] = ((features['hour'] >= 0) & (features['hour'] < 8)).astype(int)
    # Europe session (8-16 UTC)
    features['europe_session'] = ((features['hour'] >= 8) & (features['hour'] < 16)).astype(int)
    # US session (13-22 UTC)
    features['us_session'] = ((features['hour'] >= 13) & (features['hour'] < 22)).astype(int)

    # Clean data
    features = features.replace([np.inf, -np.inf], np.nan)
    features = features.fillna(0)

    return features

if __name__ == "__main__":
    test_deep_ensemble()