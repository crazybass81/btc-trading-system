#!/usr/bin/env python3
"""
Test ML Model Predictions Using Actual Model.predict()
이전 테스트는 조건 기반이었음, 이제 실제 ML 예측 사용
"""

import ccxt
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class MLPredictionTester:
    def __init__(self):
        self.exchange = ccxt.binance()
        self.models = {}
        self.load_models()

    def load_models(self):
        """Load all available ML models"""
        import glob
        model_files = glob.glob("models/specialist_*.pkl")

        for model_path in model_files:
            try:
                # Extract timeframe from filename
                filename = model_path.split('/')[-1]
                if '_combined_' in filename:
                    # e.g., specialist_15m_combined_model.pkl
                    parts = filename.replace('specialist_', '').replace('_combined_model.pkl', '')
                    timeframe = parts
                else:
                    continue

                # Load model data
                model_data = joblib.load(model_path)
                self.models[timeframe] = model_data

                # Display loaded models info
                if 'up_model' in model_data and 'down_model' in model_data:
                    print(f"✅ {timeframe} 모델 로드 (UP + DOWN 전문화)")
                else:
                    print(f"✅ {timeframe} 모델 로드")

            except Exception as e:
                print(f"⚠️ {model_path} 로드 실패: {e}")

        if not self.models:
            print("❌ 로드된 모델이 없습니다!")
            return

    def get_data(self, timeframe, days=7):
        """Get test data"""
        print(f"\n📊 {timeframe} 테스트 데이터 수집 ({days}일)...")

        tf_ms = {
            '15m': 15 * 60 * 1000,
            '30m': 30 * 60 * 1000,
            '1h': 60 * 60 * 1000,
            '4h': 4 * 60 * 60 * 1000,
        }

        ms_per_candle = tf_ms.get(timeframe, 60 * 60 * 1000)
        total_candles = int(days * 24 * 60 * 60 * 1000 / ms_per_candle) + 100  # Extra for indicators

        all_data = []
        chunk_size = 1000
        end_time = self.exchange.milliseconds()
        current_time = end_time

        while len(all_data) < total_candles:
            try:
                ohlcv = self.exchange.fetch_ohlcv(
                    'BTC/USDT',
                    timeframe,
                    limit=chunk_size,
                    since=current_time - (chunk_size * ms_per_candle)
                )

                if not ohlcv:
                    break

                all_data = ohlcv + all_data
                current_time = ohlcv[0][0] if ohlcv else current_time

                if len(all_data) >= total_candles:
                    all_data = all_data[-total_candles:]
                    break

            except:
                break

        df = pd.DataFrame(all_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)

        print(f"  ✅ {len(df)}개 캔들 수집 완료")
        return df

    def create_features(self, df, feature_names):
        """Create features matching the model's expected features"""
        features = pd.DataFrame(index=df.index)

        # Basic returns
        for period in [1, 2, 3, 5, 8, 13, 21, 34, 55, 89]:
            if f'return_{period}' in feature_names:
                features[f'return_{period}'] = df['close'].pct_change(period)
            if f'return_{period}_abs' in feature_names:
                features[f'return_{period}_abs'] = df['close'].pct_change(period).abs()

        # RSI variants
        for period in [14, 21, 28]:
            delta = df['close'].diff()
            gain = delta.where(delta > 0, 0).rolling(period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
            rsi = 100 - (100 / (1 + gain / (loss + 1e-10)))

            if f'rsi_{period}' in feature_names:
                features[f'rsi_{period}'] = rsi
            if f'rsi_{period}_ma' in feature_names:
                features[f'rsi_{period}_ma'] = rsi.rolling(10).mean()
            if f'rsi_{period}_std' in feature_names:
                features[f'rsi_{period}_std'] = rsi.rolling(10).std()

        # Moving averages and ratios
        for period in [10, 20, 50, 100, 200]:
            sma = df['close'].rolling(window=period).mean()
            ema = df['close'].ewm(span=period, adjust=False).mean()

            if f'sma_{period}' in feature_names:
                features[f'sma_{period}'] = sma
            if f'ema_{period}' in feature_names:
                features[f'ema_{period}'] = ema
            if f'sma_{period}_ratio' in feature_names:
                features[f'sma_{period}_ratio'] = df['close'] / (sma + 1e-10)
            if f'ema_{period}_ratio' in feature_names:
                features[f'ema_{period}_ratio'] = df['close'] / (ema + 1e-10)
            if f'sma_ema_diff_{period}' in feature_names:
                features[f'sma_ema_diff_{period}'] = (sma - ema) / df['close']
            if f'sma_{period}_slope' in feature_names:
                features[f'sma_{period}_slope'] = sma.pct_change(5)

        # Bollinger Bands
        for period in [20, 30]:
            ma = df['close'].rolling(window=period).mean()
            std = df['close'].rolling(window=period).std()
            if f'bb_upper_{period}' in feature_names:
                features[f'bb_upper_{period}'] = (df['close'] - (ma + 2*std)) / df['close']
            if f'bb_lower_{period}' in feature_names:
                features[f'bb_lower_{period}'] = ((ma - 2*std) - df['close']) / df['close']
            if f'bb_width_{period}' in feature_names:
                features[f'bb_width_{period}'] = (4*std) / (ma + 1e-10)
            if f'bb_position_{period}' in feature_names:
                features[f'bb_position_{period}'] = (df['close'] - ma) / (2*std + 1e-10)

        # MACD
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9, adjust=False).mean()

        if 'macd' in feature_names:
            features['macd'] = macd / (df['close'] + 1e-10)
        if 'macd_signal' in feature_names:
            features['macd_signal'] = signal / (df['close'] + 1e-10)
        if 'macd_hist' in feature_names:
            features['macd_hist'] = (macd - signal) / (df['close'] + 1e-10)

        # Volume features
        if 'volume_ratio' in feature_names:
            features['volume_ratio'] = df['volume'] / df['volume'].rolling(window=20).mean()
        if 'volume_trend' in feature_names:
            features['volume_trend'] = df['volume'].rolling(window=20).apply(
                lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0
            )
        if 'obv' in feature_names:
            obv = (df['volume'] * (~df['close'].diff().le(0) * 2 - 1)).cumsum()
            features['obv'] = obv / obv.rolling(window=20).mean()

        # VWAP
        if 'vwap' in feature_names or 'vwap_ratio' in feature_names:
            vwap = (df['close'] * df['volume']).rolling(window=20).sum() / df['volume'].rolling(window=20).sum()
            if 'vwap' in feature_names:
                features['vwap'] = vwap
            if 'vwap_ratio' in feature_names:
                features['vwap_ratio'] = df['close'] / (vwap + 1e-10)

        # Volatility
        if 'volatility_20' in feature_names:
            features['volatility_20'] = df['close'].pct_change().rolling(window=20).std()
        if 'volatility_50' in feature_names:
            features['volatility_50'] = df['close'].pct_change().rolling(window=50).std()
        if 'atr_14' in feature_names:
            high_low = df['high'] - df['low']
            high_close = (df['high'] - df['close'].shift()).abs()
            low_close = (df['low'] - df['close'].shift()).abs()
            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            features['atr_14'] = tr.rolling(window=14).mean() / df['close']

        # Price position
        for period in [20, 50, 100]:
            if f'high_{period}' in feature_names:
                features[f'high_{period}'] = df['high'].rolling(window=period).max()
            if f'low_{period}' in feature_names:
                features[f'low_{period}'] = df['low'].rolling(window=period).min()
            if f'price_position_{period}' in feature_names:
                highest = df['high'].rolling(window=period).max()
                lowest = df['low'].rolling(window=period).min()
                features[f'price_position_{period}'] = (df['close'] - lowest) / (highest - lowest + 1e-10)
            if f'dist_from_high_{period}' in feature_names:
                features[f'dist_from_high_{period}'] = (df['high'].rolling(window=period).max() - df['close']) / df['close']

        # Statistical moments
        for period in [20, 50]:
            returns = df['close'].pct_change()
            if f'return_skew_{period}' in feature_names:
                features[f'return_skew_{period}'] = returns.rolling(window=period).skew()
            if f'return_kurt_{period}' in feature_names:
                features[f'return_kurt_{period}'] = returns.rolling(window=period).kurt()
            if f'return_std_{period}' in feature_names:
                features[f'return_std_{period}'] = returns.rolling(window=period).std()

        # Time features
        if 'hour' in feature_names:
            features['hour'] = df.index.hour
        if 'day_of_week' in feature_names:
            features['day_of_week'] = df.index.dayofweek

        # Pattern features
        if 'higher_highs' in feature_names:
            features['higher_highs'] = (df['high'] > df['high'].shift(1)).rolling(5).sum()
        if 'lower_lows' in feature_names:
            features['lower_lows'] = (df['low'] < df['low'].shift(1)).rolling(5).sum()
        if 'bullish_candle' in feature_names:
            features['bullish_candle'] = ((df['close'] > df['open']) * 1.0).rolling(5).mean()
        if 'bearish_candle' in feature_names:
            features['bearish_candle'] = ((df['close'] < df['open']) * 1.0).rolling(5).mean()

        # Handle missing features (set to 0)
        for feat in feature_names:
            if feat not in features.columns:
                features[feat] = 0

        # Clean data
        features = features.replace([np.inf, -np.inf], np.nan)
        features = features.fillna(0)

        # Select only required features in correct order
        features = features[feature_names]

        return features

    def test_ml_predictions(self):
        """Test actual ML model predictions"""
        print("\n" + "="*60)
        print("🤖 실제 ML 모델 예측 테스트")
        print("="*60)

        results = {}

        for timeframe, model_data in self.models.items():
            print(f"\n{'='*60}")
            print(f"📍 {timeframe} 모델 테스트")
            print(f"{'='*60}")

            # Get test data
            df = self.get_data(timeframe, days=7)

            # Check model structure
            if 'up_model' in model_data and 'down_model' in model_data:
                # Specialist models (separate UP and DOWN)
                print(f"  📈 상승 전문 모델 테스트...")

                # Get scaler and features
                scaler = model_data.get('scaler', model_data.get('up_scaler'))
                features = model_data.get('features', model_data.get('up_features', []))

                up_results = self.test_specialist_model(
                    df, model_data['up_model'], scaler,
                    features, 'up', timeframe
                )

                print(f"\n  📉 하락 전문 모델 테스트...")
                down_results = self.test_specialist_model(
                    df, model_data['down_model'], scaler,
                    features, 'down', timeframe
                )

                results[timeframe] = {'up': up_results, 'down': down_results}

            elif 'models' in model_data:
                # Ensemble models
                print(f"  🎯 앙상블 모델 테스트...")
                ensemble_results = self.test_ensemble_model(df, model_data, timeframe)
                results[timeframe] = ensemble_results

        # Summary
        self.print_summary(results)

    def test_specialist_model(self, df, model, scaler, feature_names, direction, timeframe):
        """Test specialist model (UP or DOWN only)"""
        # Create features
        features = self.create_features(df, feature_names)

        # Remove NaN rows
        valid_idx = ~features.isna().any(axis=1)
        X = features[valid_idx]

        if len(X) < 50:
            print(f"    ⚠️ 데이터 부족: {len(X)}개")
            return None

        # Scale features
        X_scaled = scaler.transform(X)

        # Get predictions
        predictions = model.predict(X_scaled)
        pred_proba = model.predict_proba(X_scaled)[:, 1]  # Probability of positive class

        # Calculate actual direction changes
        actual_direction = []
        for i in range(len(X) - 1):
            idx = X.index[i]
            next_idx = X.index[i + 1]

            if direction == 'up':
                actual = 1 if df.loc[next_idx, 'close'] > df.loc[idx, 'close'] else 0
            else:
                actual = 1 if df.loc[next_idx, 'close'] < df.loc[idx, 'close'] else 0

            actual_direction.append(actual)

        # Trim predictions to match actual
        predictions = predictions[:-1]
        pred_proba = pred_proba[:-1]

        # Calculate metrics
        correct = sum(p == a for p, a in zip(predictions, actual_direction))
        total = len(actual_direction)
        accuracy = (correct / total * 100) if total > 0 else 0

        # Count predictions
        num_signals = sum(predictions)
        signal_accuracy = 0
        if num_signals > 0:
            signal_correct = sum(1 for i, p in enumerate(predictions) if p == 1 and actual_direction[i] == 1)
            signal_accuracy = (signal_correct / num_signals * 100)

        print(f"    예측 횟수: {num_signals}/{total} ({num_signals/total*100:.1f}%)")
        print(f"    전체 정확도: {accuracy:.1f}%")
        print(f"    신호 정확도: {signal_accuracy:.1f}%")
        print(f"    평균 확률: {np.mean(pred_proba):.3f}")

        return {
            'total': total,
            'predictions': num_signals,
            'correct': correct,
            'accuracy': accuracy,
            'signal_accuracy': signal_accuracy,
            'avg_probability': np.mean(pred_proba)
        }

    def test_ensemble_model(self, df, model_data, timeframe):
        """Test ensemble model"""
        # Get model components
        models = model_data.get('models', {})
        scaler = model_data.get('scaler')
        feature_names = model_data.get('features', [])

        if not models or not scaler:
            print(f"    ⚠️ 모델 구성 요소 부족")
            return None

        # Create features
        features = self.create_features(df, feature_names)

        # Remove NaN rows
        valid_idx = ~features.isna().any(axis=1)
        X = features[valid_idx]

        if len(X) < 50:
            print(f"    ⚠️ 데이터 부족: {len(X)}개")
            return None

        # Scale features
        X_scaled = scaler.transform(X)

        # Get predictions from each model
        all_predictions = []
        all_probabilities = []

        for model_name, model_info in models.items():
            if 'model' in model_info:
                model = model_info['model']
                try:
                    pred = model.predict(X_scaled)
                    prob = model.predict_proba(X_scaled)[:, 1]
                    all_predictions.append(pred)
                    all_probabilities.append(prob)
                    print(f"    {model_name}: {sum(pred)}/{len(pred)} signals")
                except Exception as e:
                    print(f"    {model_name} 예측 실패: {e}")

        if not all_predictions:
            print(f"    ⚠️ 예측 가능한 모델 없음")
            return None

        # Ensemble prediction (majority vote)
        ensemble_pred = np.mean(all_predictions, axis=0) > 0.5
        ensemble_prob = np.mean(all_probabilities, axis=0)

        # Calculate actual direction changes
        actual_up = []
        actual_down = []
        for i in range(len(X) - 1):
            idx = X.index[i]
            next_idx = X.index[i + 1]

            actual_up.append(1 if df.loc[next_idx, 'close'] > df.loc[idx, 'close'] else 0)
            actual_down.append(1 if df.loc[next_idx, 'close'] < df.loc[idx, 'close'] else 0)

        # Trim predictions
        ensemble_pred = ensemble_pred[:-1]
        ensemble_prob = ensemble_prob[:-1]

        # Calculate metrics
        up_correct = sum(p == a for p, a in zip(ensemble_pred, actual_up))
        down_correct = sum((1-p) == a for p, a in zip(ensemble_pred, actual_down))

        total = len(actual_up)
        up_signals = sum(ensemble_pred)
        down_signals = total - up_signals

        up_accuracy = (up_correct / total * 100) if total > 0 else 0
        down_accuracy = (down_correct / total * 100) if total > 0 else 0
        overall_accuracy = ((up_correct + down_correct) / (total * 2) * 100) if total > 0 else 0

        print(f"    📈 상승 예측: {up_signals}회, 정확도: {up_accuracy:.1f}%")
        print(f"    📉 하락 예측: {down_signals}회, 정확도: {down_accuracy:.1f}%")
        print(f"    🎯 전체 정확도: {overall_accuracy:.1f}%")
        print(f"    📊 평균 확률: {np.mean(ensemble_prob):.3f}")

        return {
            'total': total,
            'up_predictions': up_signals,
            'down_predictions': down_signals,
            'up_accuracy': up_accuracy,
            'down_accuracy': down_accuracy,
            'overall_accuracy': overall_accuracy,
            'avg_probability': np.mean(ensemble_prob)
        }

    def print_summary(self, results):
        """Print test summary"""
        print("\n" + "="*60)
        print("📊 ML 모델 예측 테스트 요약")
        print("="*60)

        for timeframe, result in results.items():
            print(f"\n{timeframe}:")

            if isinstance(result, dict) and 'up' in result:
                # Specialist models
                if result['up']:
                    print(f"  📈 UP: {result['up']['accuracy']:.1f}% (신호: {result['up']['signal_accuracy']:.1f}%)")
                if result['down']:
                    print(f"  📉 DOWN: {result['down']['accuracy']:.1f}% (신호: {result['down']['signal_accuracy']:.1f}%)")
            elif result:
                # Ensemble model
                print(f"  🎯 앙상블: {result['overall_accuracy']:.1f}%")
                print(f"     UP: {result['up_accuracy']:.1f}%, DOWN: {result['down_accuracy']:.1f}%")

        print("\n" + "="*60)
        print("💡 분석:")
        print("  - 실제 ML model.predict() 사용")
        print("  - 조건 기반이 아닌 학습된 패턴 사용")
        print("  - 60% 이상이면 사용 가능")
        print("="*60)

def main():
    print("="*60)
    print("🤖 ML 모델 실제 예측 테스트")
    print("📅 " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("="*60)

    tester = MLPredictionTester()

    if tester.models:
        tester.test_ml_predictions()
    else:
        print("❌ 테스트할 모델이 없습니다. 먼저 모델을 훈련하세요.")

if __name__ == "__main__":
    main()