#!/usr/bin/env python3
"""
Deep Ensemble 15m UP 모델 백테스트
62.8% 정확도 모델 실제 성과 테스트
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
    print("🎯 Deep Ensemble 15m UP (62.8%) 백테스트")
    print("📅 " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("="*60)

    # Load model
    model_data = joblib.load("models/deep_ensemble_15m_up_model.pkl")
    accuracy = model_data.get('ensemble_accuracy', 0) * 100
    print(f"✅ 모델 로드: {accuracy:.1f}% 훈련 정확도")
    print(f"  모델 수: {len(model_data['models'])}개")

    # Get data
    exchange = ccxt.binance()
    print("\n📊 데이터 수집...")
    ohlcv = exchange.fetch_ohlcv('BTC/USDT', '15m', limit=1000)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    print(f"  ✅ {len(df)}개 캔들 수집")
    print(f"  기간: {df.index[0].strftime('%Y-%m-%d')} ~ {df.index[-1].strftime('%Y-%m-%d')}")

    # Create features matching training
    print("\n📐 특징 생성...")
    features = pd.DataFrame(index=df.index)

    # Returns
    for period in [1, 2, 3, 5, 8, 13, 21, 34, 55]:
        features[f'return_{period}'] = df['close'].pct_change(period)
        log_ret = np.log(df['close'] / df['close'].shift(period))
        features[f'log_return_{period}'] = log_ret.replace([np.inf, -np.inf], 0)

    # RSI
    for period in [7, 14, 21, 28]:
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        features[f'rsi_{period}'] = 100 - (100 / (1 + gain / (loss + 1e-10)))
        features[f'rsi_{period}_sma'] = features[f'rsi_{period}'].rolling(5).mean()

    # SMA crosses (상승에 유리한 조합만)
    sma_pairs = [(5, 10), (5, 20), (5, 50), (5, 100), (5, 200),
                 (10, 20), (10, 50), (10, 100), (10, 200),
                 (20, 50), (20, 100), (20, 200),
                 (50, 100), (50, 200), (100, 200)]

    for short, long in sma_pairs:
        if len(df) > long:
            sma_short = df['close'].rolling(short).mean()
            sma_long = df['close'].rolling(long).mean()
            features[f'sma_cross_{short}_{long}'] = (sma_short > sma_long).astype(int)
        else:
            features[f'sma_cross_{short}_{long}'] = 0

    # Bollinger positions
    bb_params = [(10, 1.5), (10, 2), (10, 2.5),
                 (20, 1.5), (20, 2), (20, 2.5),
                 (30, 1.5), (30, 2), (30, 2.5)]

    for period, dev in bb_params:
        if len(df) > period:
            ma = df['close'].rolling(period).mean()
            std = df['close'].rolling(period).std()
            upper = ma + dev * std
            lower = ma - dev * std
            features[f'bb_pos_{period}_{dev}'] = (df['close'] - lower) / (upper - lower + 1e-10)
        else:
            features[f'bb_pos_{period}_{dev}'] = 0.5

    # Volume features
    features['volume_sma_ratio'] = df['volume'] / (df['volume'].rolling(20).mean() + 1e-10)
    features['volume_ema_ratio'] = df['volume'] / (df['volume'].ewm(span=20).mean() + 1e-10)
    features['price_volume_trend'] = (df['close'].pct_change() * df['volume']).rolling(14).sum()

    # Volatility ratios
    for period in [10, 20, 30]:
        vol_short = df['close'].pct_change().rolling(period).std()
        vol_long = df['close'].pct_change().rolling(period*2).std()
        features[f'volatility_ratio_{period}'] = vol_short / (vol_long + 1e-10)

    # High/Low features
    for period in [5, 10, 20]:
        features[f'high_low_ratio_{period}'] = df['high'].rolling(period).max() / (df['low'].rolling(period).min() + 1e-10)
        features[f'close_to_high_{period}'] = df['close'] / (df['high'].rolling(period).max() + 1e-10)
        features[f'close_to_low_{period}'] = df['close'] / (df['low'].rolling(period).min() + 1e-10)

    # UP-specific features (상승 특화)
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

    # Time features
    features['hour'] = df.index.hour
    features['is_weekend'] = (df.index.dayofweek >= 5).astype(int)

    # Clean
    features = features.replace([np.inf, -np.inf], np.nan)
    features = features.fillna(0)

    print(f"  ✅ {len(features.columns)}개 특징 생성")

    # Prepare data
    feature_names = model_data.get('features', [])
    if feature_names:
        # Add missing features
        for f in feature_names:
            if f not in features.columns:
                features[f] = 0
        X = features[feature_names]
    else:
        X = features

    # Scale data
    scaler = model_data.get('scaler')
    if scaler:
        X_scaled = scaler.transform(X)
    else:
        X_scaled = X.values

    print(f"  ✅ {X_scaled.shape[1]}개 특징 준비 완료")

    # Make predictions
    print("\n🔮 예측 수행...")
    models = model_data['models']
    weights = model_data.get('weights', {})

    all_predictions = []
    successful_models = 0

    for model_tuple in models:
        if isinstance(model_tuple, tuple) and len(model_tuple) >= 2:
            model_name, model = model_tuple[0], model_tuple[1]

            try:
                if hasattr(model, 'predict_proba'):
                    prob = model.predict_proba(X_scaled)[:, 1]
                else:
                    prob = model.predict(X_scaled)

                # Apply weight if available
                weight = weights.get(model_name, 1.0)
                all_predictions.append(prob * weight)
                successful_models += 1
            except Exception as e:
                pass

    print(f"  ✅ {successful_models}/{len(models)} 모델 예측 성공")

    if all_predictions:
        # Ensemble prediction
        ensemble_prob = np.mean(all_predictions, axis=0)
        ensemble_pred = (ensemble_prob > 0.5).astype(int)

        # Calculate accuracy
        actual_movements = []
        for i in range(len(df) - 1):
            actual = 1 if df['close'].iloc[i+1] > df['close'].iloc[i] else 0
            actual_movements.append(actual)

        predictions = ensemble_pred[:-1]
        probabilities = ensemble_prob[:-1]

        correct = sum(p == a for p, a in zip(predictions, actual_movements))
        accuracy = correct / len(actual_movements) * 100

        print(f"\n📊 백테스트 결과")
        print("="*60)
        print(f"  실제 예측 정확도: {accuracy:.1f}%")
        print(f"  UP 신호 발생: {sum(predictions)}/{len(predictions)} ({sum(predictions)/len(predictions)*100:.1f}%)")

        # Trading simulation
        trades = []
        capital = 10000
        position = 0
        confidence_threshold = 0.55  # 55% 이상 확신도만 거래

        for i in range(len(predictions)):
            if predictions[i] == 1 and probabilities[i] > confidence_threshold:
                if position == 0:
                    position = capital
                    entry_price = df['close'].iloc[i]
                    entry_time = df.index[i]

            elif position > 0:  # 1 캔들 후 청산
                exit_price = df['close'].iloc[i]
                profit = position * (exit_price - entry_price) / entry_price
                capital += profit
                trades.append({
                    'entry_time': entry_time,
                    'exit_time': df.index[i],
                    'entry': entry_price,
                    'exit': exit_price,
                    'profit': profit,
                    'return': (exit_price - entry_price) / entry_price * 100
                })
                position = 0

        # Trading results
        if trades:
            winning_trades = sum(1 for t in trades if t['profit'] > 0)
            win_rate = winning_trades / len(trades) * 100
            total_return = (capital - 10000) / 100
            avg_profit = np.mean([t['return'] for t in trades if t['profit'] > 0]) if winning_trades > 0 else 0
            avg_loss = np.mean([t['return'] for t in trades if t['profit'] <= 0]) if (len(trades) - winning_trades) > 0 else 0

            print(f"\n💰 거래 성과")
            print("-"*40)
            print(f"  총 거래 수: {len(trades)}회")
            print(f"  승률: {win_rate:.1f}%")
            print(f"  평균 수익: {avg_profit:.2f}%")
            print(f"  평균 손실: {avg_loss:.2f}%")
            print(f"  총 수익률: {total_return:.2f}%")
            print(f"  최종 자본: ${capital:.2f}")

            # Best and worst trades
            best_trade = max(trades, key=lambda x: x['return'])
            worst_trade = min(trades, key=lambda x: x['return'])

            print(f"\n📈 최고 거래: {best_trade['return']:.2f}% ({best_trade['entry_time'].strftime('%m-%d %H:%M')})")
            print(f"📉 최악 거래: {worst_trade['return']:.2f}% ({worst_trade['entry_time'].strftime('%m-%d %H:%M')})")

            # Recent trades
            print(f"\n⏰ 최근 5개 거래:")
            for trade in trades[-5:]:
                emoji = "✅" if trade['profit'] > 0 else "❌"
                print(f"  {emoji} {trade['entry_time'].strftime('%m-%d %H:%M')}: {trade['return']:.2f}%")

            # Confidence analysis
            high_conf = [p for p in probabilities if p > 0.6]
            very_high_conf = [p for p in probabilities if p > 0.65]

            print(f"\n💎 신뢰도 분석:")
            print(f"  >60% 신호: {len(high_conf)}개")
            print(f"  >65% 신호: {len(very_high_conf)}개")

            # Performance by confidence
            for threshold in [0.55, 0.60, 0.65]:
                high_conf_trades = [t for i, t in enumerate(trades) if i < len(probabilities) and probabilities[i] > threshold]
                if high_conf_trades:
                    high_conf_win = sum(1 for t in high_conf_trades if t['profit'] > 0)
                    high_conf_rate = high_conf_win / len(high_conf_trades) * 100
                    print(f"  >{threshold*100:.0f}% 신뢰도 승률: {high_conf_rate:.1f}% ({len(high_conf_trades)}개 거래)")

        else:
            print("\n⚠️ 거래 신호가 발생하지 않았습니다.")
            print(f"   신뢰도 {confidence_threshold*100:.0f}% 이상 신호가 없음")

        # Market analysis
        print(f"\n📊 시장 분석:")
        actual_up = sum(actual_movements)
        print(f"  실제 상승: {actual_up}/{len(actual_movements)} ({actual_up/len(actual_movements)*100:.1f}%)")
        print(f"  예측 상승: {sum(predictions)}/{len(predictions)} ({sum(predictions)/len(predictions)*100:.1f}%)")

        # Final verdict
        print(f"\n🎯 최종 평가:")
        if accuracy > 60:
            print(f"  ✅ 모델 성능 우수 ({accuracy:.1f}% > 60%)")
        elif accuracy > 55:
            print(f"  ⚠️ 모델 성능 보통 ({accuracy:.1f}%)")
        else:
            print(f"  ❌ 모델 성능 부족 ({accuracy:.1f}% < 55%)")

        if trades and total_return > 0:
            print(f"  💰 수익 창출 가능 (+{total_return:.2f}%)")

    else:
        print("❌ 예측 실패")

if __name__ == "__main__":
    main()