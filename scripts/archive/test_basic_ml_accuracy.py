#!/usr/bin/env python3
"""
Test Basic ML Model Accuracy
실제 model.predict() 사용하여 정확도 테스트
"""

import ccxt
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class BasicMLTester:
    def __init__(self):
        self.exchange = ccxt.binance()
        self.models = {}
        self.load_models()

    def load_models(self):
        """Load basic ML models"""
        import glob
        model_files = glob.glob("models/basic_ml_*.pkl")

        for model_path in model_files:
            try:
                # Extract timeframe
                filename = model_path.split('/')[-1]
                timeframe = filename.replace('basic_ml_', '').replace('_model.pkl', '')

                # Load model
                model_data = joblib.load(model_path)
                self.models[timeframe] = model_data

                best_acc = model_data.get('best_accuracy', 0) * 100
                ensemble_acc = model_data.get('ensemble_accuracy', 0) * 100
                auc = model_data.get('ensemble_auc', 0)

                print(f"✅ {timeframe} 모델 로드")
                print(f"   최고 정확도: {best_acc:.1f}%")
                print(f"   앙상블 정확도: {ensemble_acc:.1f}%")
                print(f"   AUC: {auc:.3f}")

            except Exception as e:
                print(f"⚠️ {model_path} 로드 실패: {e}")

    def get_realtime_data(self, timeframe, limit=100):
        """Get real-time test data"""
        print(f"\n📊 {timeframe} 실시간 데이터 수집...")

        ohlcv = self.exchange.fetch_ohlcv('BTC/USDT', timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)

        print(f"  ✅ {len(df)}개 캔들 수집")
        return df

    def create_features(self, df, feature_names):
        """Create features matching model requirements"""
        features = pd.DataFrame(index=df.index)

        # Create all features from training
        # Price returns
        for period in [1, 3, 5, 10, 20]:
            features[f'return_{period}'] = df['close'].pct_change(period)

        # RSI
        for period in [14, 21]:
            delta = df['close'].diff()
            gain = delta.where(delta > 0, 0).rolling(period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
            features[f'rsi_{period}'] = 100 - (100 / (1 + gain / (loss + 1e-10)))

        # Moving averages
        for period in [10, 20, 50]:
            ma = df['close'].rolling(window=period).mean()
            features[f'ma_{period}_ratio'] = (df['close'] - ma) / (ma + 1e-10)

        # Bollinger Bands
        period = 20
        ma = df['close'].rolling(window=period).mean()
        std = df['close'].rolling(window=period).std()
        features['bb_upper'] = (df['close'] - (ma + 2*std)) / df['close']
        features['bb_lower'] = ((ma - 2*std) - df['close']) / df['close']
        features['bb_width'] = (4*std) / (ma + 1e-10)

        # MACD
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9, adjust=False).mean()
        features['macd'] = macd / (df['close'] + 1e-10)
        features['macd_signal'] = signal / (df['close'] + 1e-10)
        features['macd_hist'] = (macd - signal) / (df['close'] + 1e-10)

        # Volume
        features['volume_ratio'] = df['volume'] / df['volume'].rolling(window=20).mean()

        # Volatility
        features['volatility'] = df['close'].pct_change().rolling(window=20).std()

        # Price position
        highest_20 = df['high'].rolling(window=20).max()
        lowest_20 = df['low'].rolling(window=20).min()
        features['price_position'] = (df['close'] - lowest_20) / (highest_20 - lowest_20 + 1e-10)

        # Time features
        features['hour'] = df.index.hour
        features['day_of_week'] = df.index.dayofweek

        # Clean and select features
        features = features.replace([np.inf, -np.inf], np.nan)
        features = features.fillna(0)
        features = features[feature_names]

        return features

    def test_predictions(self):
        """Test model predictions on real-time data"""
        print("\n" + "="*60)
        print("🎯 실시간 ML 예측 테스트")
        print("="*60)

        results = {}

        for timeframe, model_data in self.models.items():
            print(f"\n📍 {timeframe} 모델 테스트")
            print("-"*40)

            # Get real-time data
            df = self.get_realtime_data(timeframe, limit=100)

            # Get model components
            models = model_data.get('models', {})
            scaler = model_data.get('scaler')
            feature_names = model_data.get('features', [])

            # Create features
            features = self.create_features(df, feature_names)

            # Remove NaN
            valid_idx = ~features.isna().any(axis=1)
            X = features[valid_idx]

            if len(X) < 10:
                print(f"  ⚠️ 데이터 부족")
                continue

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
                    except:
                        continue

            # Ensemble prediction
            if all_predictions:
                ensemble_prob = np.mean(all_probabilities, axis=0)
                ensemble_pred = (ensemble_prob > 0.5).astype(int)

                # Latest predictions
                latest_pred = ensemble_pred[-5:]
                latest_prob = ensemble_prob[-5:]
                latest_times = X.index[-5:]

                print(f"\n  📈 최근 5개 예측:")
                for i, (time, pred, prob) in enumerate(zip(latest_times, latest_pred, latest_prob)):
                    direction = "UP" if pred == 1 else "DOWN"
                    confidence = prob if pred == 1 else (1 - prob)
                    emoji = "📈" if pred == 1 else "📉"
                    print(f"    {time.strftime('%H:%M')}: {emoji} {direction} (신뢰도: {confidence:.1%})")

                # Overall statistics
                up_predictions = sum(ensemble_pred)
                down_predictions = len(ensemble_pred) - up_predictions
                avg_confidence = np.mean(np.maximum(ensemble_prob, 1 - ensemble_prob))

                print(f"\n  📊 전체 통계:")
                print(f"    UP 예측: {up_predictions}/{len(ensemble_pred)} ({up_predictions/len(ensemble_pred)*100:.1f}%)")
                print(f"    DOWN 예측: {down_predictions}/{len(ensemble_pred)} ({down_predictions/len(ensemble_pred)*100:.1f}%)")
                print(f"    평균 신뢰도: {avg_confidence:.1%}")

                results[timeframe] = {
                    'up_count': up_predictions,
                    'down_count': down_predictions,
                    'total': len(ensemble_pred),
                    'avg_confidence': avg_confidence,
                    'latest_prediction': "UP" if latest_pred[-1] == 1 else "DOWN",
                    'latest_confidence': latest_prob[-1] if latest_pred[-1] == 1 else (1 - latest_prob[-1])
                }

        # Summary
        self.print_summary(results)
        return results

    def backtest_accuracy(self, days=3):
        """Backtest model accuracy on historical data"""
        print("\n" + "="*60)
        print(f"📊 {days}일 백테스트")
        print("="*60)

        backtest_results = {}

        for timeframe, model_data in self.models.items():
            print(f"\n{timeframe} 백테스트:")

            # Get more historical data
            limit = days * 24 * 60 // {'15m': 15, '30m': 30, '1h': 60, '4h': 240}.get(timeframe, 60)
            df = self.get_realtime_data(timeframe, limit=min(limit, 1000))

            # Get model components
            models = model_data.get('models', {})
            scaler = model_data.get('scaler')
            feature_names = model_data.get('features', [])

            # Create features
            features = self.create_features(df, feature_names)

            # Remove NaN
            valid_idx = ~features.isna().any(axis=1)
            X = features[valid_idx]
            df_valid = df[valid_idx]

            if len(X) < 50:
                print(f"  ⚠️ 데이터 부족")
                continue

            # Scale features
            X_scaled = scaler.transform(X)

            # Get ensemble predictions
            all_probabilities = []
            for model_name, model_info in models.items():
                if 'model' in model_info:
                    try:
                        prob = model_info['model'].predict_proba(X_scaled)[:, 1]
                        all_probabilities.append(prob)
                    except:
                        continue

            if all_probabilities:
                ensemble_prob = np.mean(all_probabilities, axis=0)
                ensemble_pred = (ensemble_prob > 0.5).astype(int)

                # Calculate actual movements (next candle)
                actual_movements = []
                for i in range(len(df_valid) - 1):
                    actual = 1 if df_valid['close'].iloc[i+1] > df_valid['close'].iloc[i] else 0
                    actual_movements.append(actual)

                # Trim predictions to match
                predictions = ensemble_pred[:-1]
                probabilities = ensemble_prob[:-1]

                # Calculate accuracy
                correct = sum(p == a for p, a in zip(predictions, actual_movements))
                accuracy = (correct / len(actual_movements) * 100) if actual_movements else 0

                # Calculate accuracy for high confidence predictions
                high_conf_threshold = 0.6
                high_conf_idx = np.where(np.maximum(probabilities, 1 - probabilities) > high_conf_threshold)[0]
                if len(high_conf_idx) > 0:
                    high_conf_correct = sum(predictions[i] == actual_movements[i] for i in high_conf_idx)
                    high_conf_accuracy = high_conf_correct / len(high_conf_idx) * 100
                else:
                    high_conf_accuracy = 0

                print(f"  전체 정확도: {accuracy:.1f}% ({correct}/{len(actual_movements)})")
                print(f"  높은 신뢰도(>{high_conf_threshold:.0%}) 정확도: {high_conf_accuracy:.1f}% ({len(high_conf_idx)}개)")

                backtest_results[timeframe] = {
                    'accuracy': accuracy,
                    'high_conf_accuracy': high_conf_accuracy,
                    'total_predictions': len(actual_movements),
                    'high_conf_count': len(high_conf_idx)
                }

        return backtest_results

    def print_summary(self, results):
        """Print test summary"""
        print("\n" + "="*60)
        print("📋 테스트 요약")
        print("="*60)

        if results:
            print("\n최신 예측:")
            for tf, res in results.items():
                emoji = "📈" if res['latest_prediction'] == "UP" else "📉"
                print(f"  {tf}: {emoji} {res['latest_prediction']} (신뢰도: {res['latest_confidence']:.1%})")

            print("\n신뢰도 통계:")
            for tf, res in results.items():
                print(f"  {tf}: 평균 {res['avg_confidence']:.1%}")

def main():
    print("="*60)
    print("🤖 Basic ML 모델 실시간 테스트")
    print("📅 " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("="*60)

    tester = BasicMLTester()

    if tester.models:
        # Test real-time predictions
        results = tester.test_predictions()

        # Run backtest
        backtest_results = tester.backtest_accuracy(days=3)

        # Final summary
        print("\n" + "="*60)
        print("🎯 최종 평가")
        print("="*60)

        for tf in tester.models.keys():
            print(f"\n{tf}:")
            if tf in backtest_results:
                acc = backtest_results[tf]['accuracy']
                high_acc = backtest_results[tf]['high_conf_accuracy']

                if acc >= 60:
                    print(f"  ✅ 사용 가능 (정확도: {acc:.1f}%)")
                elif acc >= 55:
                    print(f"  ⚠️ 개선 필요 (정확도: {acc:.1f}%)")
                else:
                    print(f"  ❌ 재훈련 필요 (정확도: {acc:.1f}%)")

                if high_acc >= 65:
                    print(f"  💎 높은 신뢰도 신호 우수: {high_acc:.1f}%")

    else:
        print("❌ 테스트할 모델이 없습니다.")

if __name__ == "__main__":
    main()