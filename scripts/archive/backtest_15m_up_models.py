#!/usr/bin/env python3
"""
백테스트: Advanced ML 15m UP (65.2%) vs Deep Ensemble 15m UP (62.8%)
실제 거래 성과 테스트
"""

import ccxt
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class BacktestManager:
    def __init__(self):
        self.exchange = ccxt.binance()
        self.models = {}
        self.load_models()

    def load_models(self):
        """Load successful UP models"""
        print("="*60)
        print("📊 15분 UP 모델 로드")
        print("="*60)

        # Load Advanced ML model
        try:
            adv_model = joblib.load("models/advanced_15m_up_model.pkl")
            self.models['advanced_ml'] = {
                'data': adv_model,
                'accuracy': adv_model.get('best_accuracy', 0) * 100,
                'ensemble_accuracy': adv_model.get('ensemble_accuracy', 0) * 100
            }
            print(f"✅ Advanced ML 15m UP: {self.models['advanced_ml']['accuracy']:.1f}%")
        except Exception as e:
            print(f"❌ Advanced ML 로드 실패: {e}")

        # Load Deep Ensemble model
        try:
            deep_model = joblib.load("models/deep_ensemble_15m_up_model.pkl")
            self.models['deep_ensemble'] = {
                'data': deep_model,
                'accuracy': deep_model.get('ensemble_accuracy', 0) * 100
            }
            print(f"✅ Deep Ensemble 15m UP: {self.models['deep_ensemble']['accuracy']:.1f}%")
        except Exception as e:
            print(f"❌ Deep Ensemble 로드 실패: {e}")

    def get_historical_data(self, days=7):
        """Get historical data for backtesting"""
        print(f"\n📊 {days}일 데이터 수집...")

        # Calculate number of candles needed
        candles_per_day = 96  # 15분 캔들
        limit = days * candles_per_day

        # Fetch data
        ohlcv = self.exchange.fetch_ohlcv('BTC/USDT', '15m', limit=min(limit, 1000))
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)

        print(f"  ✅ {len(df)}개 캔들 수집 ({df.index[0].strftime('%Y-%m-%d')} ~ {df.index[-1].strftime('%Y-%m-%d')})")
        return df

    def create_features_advanced(self, df):
        """Create features for Advanced ML model"""
        features = pd.DataFrame(index=df.index)

        # Basic returns
        for period in [1, 2, 3, 5, 8, 13, 21, 34, 55, 89]:
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

        # Clean
        features = features.replace([np.inf, -np.inf], np.nan)
        features = features.fillna(0)

        return features

    def create_features_deep_ensemble(self, df):
        """Create features for Deep Ensemble model"""
        features = pd.DataFrame(index=df.index)

        # Returns (regular and log)
        for period in [1, 2, 3, 5, 8, 13, 21, 34, 55]:
            features[f'return_{period}'] = df['close'].pct_change(period)
            features[f'log_return_{period}'] = np.log(df['close'] / df['close'].shift(period))

        # RSI variations
        for period in [7, 14, 21, 28]:
            delta = df['close'].diff()
            gain = delta.where(delta > 0, 0).rolling(period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
            features[f'rsi_{period}'] = 100 - (100 / (1 + gain / (loss + 1e-10)))
            features[f'rsi_{period}_sma'] = features[f'rsi_{period}'].rolling(5).mean()

        # SMA crosses
        sma_periods = [5, 10, 20, 50, 100, 200]
        for i in range(len(sma_periods)):
            for j in range(i+1, len(sma_periods)):
                short = sma_periods[i]
                long = sma_periods[j]
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

        # Volume features
        features['volume_sma_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
        features['volume_ema_ratio'] = df['volume'] / df['volume'].ewm(span=20).mean()
        features['price_volume_trend'] = (df['close'].pct_change() * df['volume']).rolling(14).sum()

        # Volatility ratios
        for period in [10, 20, 30]:
            features[f'volatility_ratio_{period}'] = (df['close'].pct_change().rolling(period).std() /
                                                       df['close'].pct_change().rolling(period*2).std())

        # High/Low features
        for period in [5, 10, 20]:
            features[f'high_low_ratio_{period}'] = df['high'].rolling(period).max() / df['low'].rolling(period).min()
            features[f'close_to_high_{period}'] = df['close'] / df['high'].rolling(period).max()
            features[f'close_to_low_{period}'] = df['close'] / df['low'].rolling(period).min()

        # UP-specific features
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

        return features

    def predict_advanced_ml(self, features):
        """Predict using Advanced ML model"""
        model_data = self.models['advanced_ml']['data']
        models = model_data.get('models', {})
        scaler = model_data.get('scaler')
        selector = model_data.get('selector')
        selected_features = model_data.get('selected_features', [])

        # Select features
        if selected_features and all(f in features.columns for f in selected_features):
            X = features[selected_features]
        else:
            X = features

        # Scale
        if scaler:
            X_scaled = scaler.transform(X)
        else:
            X_scaled = X.values

        # Feature selection
        if selector:
            X_selected = selector.transform(X_scaled)
        else:
            X_selected = X_scaled

        # Predict with each model
        predictions = []
        for model_name, model_info in models.items():
            if isinstance(model_info, dict) and 'model' in model_info:
                model = model_info['model']
                try:
                    if hasattr(model, 'predict_proba'):
                        prob = model.predict_proba(X_selected)[:, 1]
                    else:
                        prob = model.predict(X_selected)
                    predictions.append(prob)
                except:
                    pass

        # Ensemble
        if predictions:
            ensemble_prob = np.mean(predictions, axis=0)
            return (ensemble_prob > 0.5).astype(int), ensemble_prob
        else:
            return np.zeros(len(X)), np.zeros(len(X))

    def predict_deep_ensemble(self, features):
        """Predict using Deep Ensemble model"""
        model_data = self.models['deep_ensemble']['data']
        models = model_data.get('models', [])  # List of models
        scalers = model_data.get('scalers', {})
        feature_names = model_data.get('features', [])
        weights = model_data.get('weights', {})

        # Select features
        if feature_names and all(f in features.columns for f in feature_names):
            X = features[feature_names]
        else:
            X = features

        # Predict with each model
        predictions = []
        for i, model_info in enumerate(models):
            if isinstance(model_info, dict) and 'model' in model_info:
                model = model_info['model']
                model_name = model_info.get('name', f'model_{i}')

                # Get scaler
                scaler = scalers.get(model_name)
                if scaler:
                    X_scaled = scaler.transform(X)
                else:
                    X_scaled = X.values

                # Predict
                try:
                    if hasattr(model, 'predict_proba'):
                        prob = model.predict_proba(X_scaled)[:, 1]
                    else:
                        prob = model.predict(X_scaled)

                    # Apply weight
                    weight = weights.get(model_name, 1.0)
                    predictions.append(prob * weight)
                except:
                    pass

        # Ensemble
        if predictions:
            ensemble_prob = np.mean(predictions, axis=0)
            return (ensemble_prob > 0.5).astype(int), ensemble_prob
        else:
            return np.zeros(len(X)), np.zeros(len(X))

    def backtest(self, df, model_type='both'):
        """Run backtest on historical data"""
        results = {}

        # Advanced ML backtest
        if model_type in ['both', 'advanced_ml'] and 'advanced_ml' in self.models:
            print("\n📊 Advanced ML (65.2%) 백테스트")
            print("-"*40)

            features = self.create_features_advanced(df)
            valid_idx = ~features.isna().any(axis=1)
            features_valid = features[valid_idx]
            df_valid = df[valid_idx]

            predictions, probabilities = self.predict_advanced_ml(features_valid)

            # Calculate performance
            actual_movements = []
            for i in range(len(df_valid) - 1):
                actual = 1 if df_valid['close'].iloc[i+1] > df_valid['close'].iloc[i] else 0
                actual_movements.append(actual)

            predictions = predictions[:-1]
            probabilities = probabilities[:-1]

            # Accuracy
            correct = sum(p == a for p, a in zip(predictions, actual_movements))
            accuracy = correct / len(actual_movements) * 100 if actual_movements else 0

            # Trading simulation
            trades = []
            capital = 10000
            position = 0

            for i in range(len(predictions)):
                if predictions[i] == 1 and probabilities[i] > 0.55:  # UP signal with confidence
                    if position == 0:
                        position = capital
                        entry_price = df_valid['close'].iloc[i]
                        entry_time = df_valid.index[i]

                elif position > 0 and i > 0:  # Close position after 1 candle
                    exit_price = df_valid['close'].iloc[i]
                    profit = position * (exit_price - entry_price) / entry_price
                    capital += profit
                    trades.append({
                        'entry_time': entry_time,
                        'exit_time': df_valid.index[i],
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'profit': profit,
                        'return': (exit_price - entry_price) / entry_price * 100
                    })
                    position = 0

            # Results
            total_trades = len(trades)
            winning_trades = sum(1 for t in trades if t['profit'] > 0)
            win_rate = winning_trades / total_trades * 100 if total_trades > 0 else 0
            total_return = (capital - 10000) / 100  # Percentage

            print(f"  예측 정확도: {accuracy:.1f}%")
            print(f"  총 거래 수: {total_trades}")
            print(f"  승률: {win_rate:.1f}%")
            print(f"  총 수익률: {total_return:.2f}%")

            results['advanced_ml'] = {
                'accuracy': accuracy,
                'trades': total_trades,
                'win_rate': win_rate,
                'total_return': total_return,
                'trade_details': trades
            }

        # Deep Ensemble backtest
        if model_type in ['both', 'deep_ensemble'] and 'deep_ensemble' in self.models:
            print("\n📊 Deep Ensemble (62.8%) 백테스트")
            print("-"*40)

            features = self.create_features_deep_ensemble(df)
            valid_idx = ~features.isna().any(axis=1)
            features_valid = features[valid_idx]
            df_valid = df[valid_idx]

            predictions, probabilities = self.predict_deep_ensemble(features_valid)

            # Calculate performance
            actual_movements = []
            for i in range(len(df_valid) - 1):
                actual = 1 if df_valid['close'].iloc[i+1] > df_valid['close'].iloc[i] else 0
                actual_movements.append(actual)

            predictions = predictions[:-1]
            probabilities = probabilities[:-1]

            # Accuracy
            correct = sum(p == a for p, a in zip(predictions, actual_movements))
            accuracy = correct / len(actual_movements) * 100 if actual_movements else 0

            # Trading simulation
            trades = []
            capital = 10000
            position = 0

            for i in range(len(predictions)):
                if predictions[i] == 1 and probabilities[i] > 0.55:  # UP signal with confidence
                    if position == 0:
                        position = capital
                        entry_price = df_valid['close'].iloc[i]
                        entry_time = df_valid.index[i]

                elif position > 0 and i > 0:  # Close position after 1 candle
                    exit_price = df_valid['close'].iloc[i]
                    profit = position * (exit_price - entry_price) / entry_price
                    capital += profit
                    trades.append({
                        'entry_time': entry_time,
                        'exit_time': df_valid.index[i],
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'profit': profit,
                        'return': (exit_price - entry_price) / entry_price * 100
                    })
                    position = 0

            # Results
            total_trades = len(trades)
            winning_trades = sum(1 for t in trades if t['profit'] > 0)
            win_rate = winning_trades / total_trades * 100 if total_trades > 0 else 0
            total_return = (capital - 10000) / 100  # Percentage

            print(f"  예측 정확도: {accuracy:.1f}%")
            print(f"  총 거래 수: {total_trades}")
            print(f"  승률: {win_rate:.1f}%")
            print(f"  총 수익률: {total_return:.2f}%")

            results['deep_ensemble'] = {
                'accuracy': accuracy,
                'trades': total_trades,
                'win_rate': win_rate,
                'total_return': total_return,
                'trade_details': trades
            }

        return results

    def print_summary(self, results):
        """Print backtest summary"""
        print("\n" + "="*60)
        print("📋 백테스트 요약")
        print("="*60)

        if 'advanced_ml' in results and 'deep_ensemble' in results:
            adv = results['advanced_ml']
            deep = results['deep_ensemble']

            print("\n📊 모델 비교:")
            print(f"  Advanced ML (65.2% 훈련):")
            print(f"    - 실제 정확도: {adv['accuracy']:.1f}%")
            print(f"    - 승률: {adv['win_rate']:.1f}%")
            print(f"    - 수익률: {adv['total_return']:.2f}%")

            print(f"\n  Deep Ensemble (62.8% 훈련):")
            print(f"    - 실제 정확도: {deep['accuracy']:.1f}%")
            print(f"    - 승률: {deep['win_rate']:.1f}%")
            print(f"    - 수익률: {deep['total_return']:.2f}%")

            # Winner
            print("\n🏆 우승 모델:")
            if adv['total_return'] > deep['total_return']:
                print(f"  Advanced ML - 수익률 {adv['total_return']:.2f}%")
            else:
                print(f"  Deep Ensemble - 수익률 {deep['total_return']:.2f}%")

def main():
    print("="*60)
    print("🎯 15분 UP 모델 백테스트")
    print("📅 " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("="*60)

    manager = BacktestManager()

    if manager.models:
        # Get historical data
        df = manager.get_historical_data(days=7)

        # Run backtest
        results = manager.backtest(df, model_type='both')

        # Print summary
        manager.print_summary(results)

        # Show recent trades
        print("\n📈 최근 거래 예시:")
        for model_name, model_results in results.items():
            if model_results['trade_details']:
                print(f"\n{model_name} 최근 3개 거래:")
                for trade in model_results['trade_details'][-3:]:
                    print(f"  {trade['entry_time'].strftime('%m-%d %H:%M')} → {trade['exit_time'].strftime('%H:%M')}: {trade['return']:.2f}%")
    else:
        print("❌ 로드된 모델이 없습니다.")

if __name__ == "__main__":
    main()