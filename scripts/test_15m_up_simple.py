#!/usr/bin/env python3
"""
간단한 15분 UP 모델 백테스트
Deep Ensemble 모델만 테스트 (더 안정적)
"""

import ccxt
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def main():
    print("="*60)
    print("🎯 Deep Ensemble 15m UP 모델 백테스트")
    print("📅 " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("="*60)

    # Load model
    model_data = joblib.load("models/deep_ensemble_15m_up_model.pkl")
    accuracy = model_data.get('ensemble_accuracy', 0) * 100
    print(f"✅ 모델 로드: {accuracy:.1f}% 정확도")

    # Get data
    exchange = ccxt.binance()
    print("\n📊 데이터 수집 중...")
    ohlcv = exchange.fetch_ohlcv('BTC/USDT', '15m', limit=1000)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    print(f"  ✅ {len(df)}개 캔들 수집")

    # Create simple features
    features = pd.DataFrame(index=df.index)

    # Basic returns
    for period in [1, 2, 3, 5, 8, 13, 21, 34, 55]:
        features[f'return_{period}'] = df['close'].pct_change(period)
        features[f'log_return_{period}'] = np.log(df['close'] / df['close'].shift(period))

    # RSI
    for period in [7, 14, 21, 28]:
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        features[f'rsi_{period}'] = 100 - (100 / (1 + gain / (loss + 1e-10)))
        features[f'rsi_{period}_sma'] = features[f'rsi_{period}'].rolling(5).mean()

    # Time
    features['hour'] = df.index.hour
    features['is_weekend'] = (df.index.dayofweek >= 5).astype(int)

    # Add more features to match model requirements
    # SMA crosses
    for short, long in [(5, 10), (5, 20), (5, 50), (5, 100), (5, 200),
                        (10, 20), (10, 50), (10, 100), (10, 200),
                        (20, 50), (20, 100), (20, 200),
                        (50, 100), (50, 200), (100, 200)]:
        if len(df) > long:
            sma_short = df['close'].rolling(short).mean()
            sma_long = df['close'].rolling(long).mean()
            features[f'sma_cross_{short}_{long}'] = (sma_short > sma_long).astype(int)

    # Bollinger positions
    for period in [10, 20, 30]:
        for dev in [1.5, 2, 2.5]:
            ma = df['close'].rolling(period).mean()
            std = df['close'].rolling(period).std()
            upper = ma + dev * std
            lower = ma - dev * std
            features[f'bb_pos_{period}_{dev}'] = (df['close'] - lower) / (upper - lower + 1e-10)

    # Volume
    features['volume_sma_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
    features['volume_ema_ratio'] = df['volume'] / df['volume'].ewm(span=20).mean()
    features['price_volume_trend'] = (df['close'].pct_change() * df['volume']).rolling(14).sum()

    # Volatility
    for period in [10, 20, 30]:
        features[f'volatility_ratio_{period}'] = (df['close'].pct_change().rolling(period).std() /
                                                  df['close'].pct_change().rolling(period*2).std())

    # High/Low
    for period in [5, 10, 20]:
        features[f'high_low_ratio_{period}'] = df['high'].rolling(period).max() / df['low'].rolling(period).min()
        features[f'close_to_high_{period}'] = df['close'] / df['high'].rolling(period).max()
        features[f'close_to_low_{period}'] = df['close'] / df['low'].rolling(period).min()

    # Special UP features
    features['micro_momentum'] = df['close'].pct_change(1).rolling(3).mean()
    features['quick_reversal'] = ((df['low'].shift(1) < df['low'].shift(2)) &
                                  (df['close'] > df['open'])).astype(int)
    features['volume_burst'] = (df['volume'] > df['volume'].rolling(10).mean() * 1.5).astype(int)
    features['bullish_pressure'] = ((df['close'] - df['low']) / (df['high'] - df['low'] + 1e-10)).rolling(5).mean()
    features['higher_highs'] = ((df['high'] > df['high'].shift(1)) &
                               (df['low'] > df['low'].shift(1))).astype(int).rolling(3).sum()
    features['dip_buying'] = ((df['low'] < df['low'].rolling(10).min()) &
                             (df['close'] > df['open'])).astype(int)
    features['accumulation'] = ((df['volume'] > df['volume'].rolling(20).mean()) &
                               (df['close'] > df['open'])).astype(int).rolling(5).sum()

    # Clean
    features = features.replace([np.inf, -np.inf], np.nan)
    features = features.fillna(0)

    print(f"  📐 {len(features.columns)}개 특징 생성")

    # Get model components
    models = model_data.get('models', [])
    scalers = model_data.get('scalers', {})
    feature_names = model_data.get('features', [])
    weights = model_data.get('weights', {})

    # Select features
    if feature_names:
        missing = [f for f in feature_names if f not in features.columns]
        if missing:
            print(f"  ⚠️ {len(missing)}개 특징 누락, 0으로 채움")
            for f in missing:
                features[f] = 0
        X = features[feature_names]
    else:
        X = features

    # Remove NaN
    valid_idx = ~X.isna().any(axis=1)
    X_valid = X[valid_idx]
    df_valid = df[valid_idx]

    print(f"  ✅ {len(X_valid)}개 유효 샘플")

    # Predict
    print("\n🔮 예측 수행 중...")
    all_predictions = []

    for i, model_info in enumerate(models):
        if isinstance(model_info, dict) and 'model' in model_info:
            model = model_info['model']
            model_name = model_info.get('name', f'model_{i}')

            # Get scaler
            scaler = scalers.get(model_name)
            if scaler:
                try:
                    X_scaled = scaler.transform(X_valid)
                except:
                    X_scaled = X_valid.values
            else:
                X_scaled = X_valid.values

            # Predict
            try:
                if hasattr(model, 'predict_proba'):
                    prob = model.predict_proba(X_scaled)[:, 1]
                else:
                    prob = model.predict(X_scaled)

                # Apply weight
                weight = weights.get(model_name, 1.0)
                all_predictions.append(prob * weight)
                print(f"  ✅ {model_name}: 예측 완료")
            except Exception as e:
                print(f"  ❌ {model_name}: {e}")

    # Ensemble
    if all_predictions:
        ensemble_prob = np.mean(all_predictions, axis=0)
        ensemble_pred = (ensemble_prob > 0.5).astype(int)

        print(f"\n📊 백테스트 결과:")
        print("-"*40)

        # Calculate actual movements
        actual_movements = []
        for i in range(len(df_valid) - 1):
            actual = 1 if df_valid['close'].iloc[i+1] > df_valid['close'].iloc[i] else 0
            actual_movements.append(actual)

        # Trim predictions
        predictions = ensemble_pred[:-1]
        probabilities = ensemble_prob[:-1]

        # Calculate accuracy
        correct = sum(p == a for p, a in zip(predictions, actual_movements))
        accuracy = correct / len(actual_movements) * 100

        print(f"  예측 정확도: {accuracy:.1f}%")
        print(f"  UP 예측 수: {sum(predictions)}/{len(predictions)}")

        # Trading simulation
        trades = []
        capital = 10000
        position = 0

        for i in range(len(predictions)):
            if predictions[i] == 1 and probabilities[i] > 0.55:  # UP signal
                if position == 0:
                    position = capital
                    entry_price = df_valid['close'].iloc[i]
                    entry_time = df_valid.index[i]

            elif position > 0 and i > 0:  # Close after 1 candle
                exit_price = df_valid['close'].iloc[i]
                profit = position * (exit_price - entry_price) / entry_price
                capital += profit
                trades.append({
                    'entry_time': entry_time,
                    'exit_time': df_valid.index[i],
                    'profit': profit,
                    'return': (exit_price - entry_price) / entry_price * 100
                })
                position = 0

        # Results
        if trades:
            total_trades = len(trades)
            winning_trades = sum(1 for t in trades if t['profit'] > 0)
            win_rate = winning_trades / total_trades * 100
            total_return = (capital - 10000) / 100

            print(f"\n💰 거래 성과:")
            print(f"  총 거래 수: {total_trades}")
            print(f"  승률: {win_rate:.1f}%")
            print(f"  총 수익률: {total_return:.2f}%")
            print(f"  최종 자본: ${capital:.2f}")

            # Recent trades
            print(f"\n📈 최근 5개 거래:")
            for trade in trades[-5:]:
                status = "✅" if trade['profit'] > 0 else "❌"
                print(f"  {status} {trade['entry_time'].strftime('%m-%d %H:%M')}: {trade['return']:.2f}%")

            # High confidence trades
            high_conf = [(i, p) for i, p in enumerate(probabilities) if p > 0.6 and predictions[i] == 1]
            if high_conf:
                print(f"\n💎 높은 신뢰도 신호 (>60%): {len(high_conf)}개")

        else:
            print("  ⚠️ 거래 신호가 없습니다")

    else:
        print("❌ 예측 실패")

if __name__ == "__main__":
    main()