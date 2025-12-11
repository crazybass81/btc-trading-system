#!/usr/bin/env python3
"""
전문 모델 예측 시스템
각 타임프레임별 상승/하락 확률 제공
"""

import pandas as pd
import numpy as np
import ccxt
import joblib
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class SpecialistPredictor:
    """상승/하락 전문 모델 예측기"""

    def __init__(self):
        self.exchange = ccxt.binance()
        self.models = {}
        self.load_models()

    def load_models(self):
        """전문 모델 로드"""
        model_dir = os.path.join(os.path.dirname(__file__), '..', 'models')

        timeframes = ['15m', '30m', '1h', '4h']

        for timeframe in timeframes:
            model_path = os.path.join(model_dir, f'specialist_{timeframe}_combined_model.pkl')

            if os.path.exists(model_path):
                try:
                    self.models[timeframe] = joblib.load(model_path)
                    print(f"✅ {timeframe} 전문 모델 로드 완료")
                except Exception as e:
                    print(f"❌ {timeframe} 모델 로드 실패: {e}")
            else:
                print(f"⚠️ {timeframe} 모델 파일 없음")

    def create_specialized_features(self, df, direction='up'):
        """방향별 특화 특징 생성"""
        features = pd.DataFrame(index=df.index)

        # 공통 특징
        for period in [1, 3, 5, 10, 20]:
            features[f'return_{period}'] = df['close'].pct_change(period)

        # RSI
        for period in [7, 14, 21]:
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / (loss + 1e-10)
            features[f'rsi_{period}'] = 100 - (100 / (1 + rs))

        # 이동평균
        for period in [10, 20, 50, 100]:
            ma = df['close'].rolling(window=period).mean()
            features[f'ma_{period}_ratio'] = (df['close'] - ma) / ma

        # MACD
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9, adjust=False).mean()
        features['macd'] = macd / df['close']
        features['macd_signal'] = signal / df['close']
        features['macd_hist'] = (macd - signal) / df['close']

        # 볼린저 밴드
        for period in [20, 50]:
            ma = df['close'].rolling(window=period).mean()
            std = df['close'].rolling(window=period).std()
            features[f'bb_{period}_upper'] = (df['close'] - (ma + 2*std)) / df['close']
            features[f'bb_{period}_lower'] = ((ma - 2*std) - df['close']) / df['close']
            features[f'bb_{period}_width'] = (4 * std) / ma

        # 거래량
        features['volume_ratio'] = df['volume'] / df['volume'].rolling(window=20).mean()
        features['volume_change'] = df['volume'].pct_change()

        if direction == 'up':
            # 상승 특화 특징
            features['support_break'] = df['close'] / df['low'].rolling(window=20).min() - 1
            features['up_momentum'] = (df['close'] > df['close'].shift(1)).rolling(window=10).sum()
            features['new_high_20'] = (df['high'] == df['high'].rolling(window=20).max()).astype(int)
            features['new_high_50'] = (df['high'] == df['high'].rolling(window=50).max()).astype(int)
            features['bullish_ratio'] = ((df['close'] > df['open']).rolling(window=10).sum()) / 10
            up_days = df['close'] > df['close'].shift(1)
            features['up_volume'] = (df['volume'] * up_days).rolling(window=10).sum()
            ma50 = df['close'].rolling(window=50).mean()
            ma200 = df['close'].rolling(window=200).mean()
            features['golden_cross'] = ((ma50 > ma200) & (ma50.shift(1) <= ma200.shift(1))).astype(int)

        else:  # down
            # 하락 특화 특징
            features['resistance_break'] = 1 - df['close'] / df['high'].rolling(window=20).max()
            features['down_momentum'] = (df['close'] < df['close'].shift(1)).rolling(window=10).sum()
            features['new_low_20'] = (df['low'] == df['low'].rolling(window=20).min()).astype(int)
            features['new_low_50'] = (df['low'] == df['low'].rolling(window=50).min()).astype(int)
            features['bearish_ratio'] = ((df['close'] < df['open']).rolling(window=10).sum()) / 10
            down_days = df['close'] < df['close'].shift(1)
            features['down_volume'] = (df['volume'] * down_days).rolling(window=10).sum()
            ma50 = df['close'].rolling(window=50).mean()
            ma200 = df['close'].rolling(window=200).mean()
            features['death_cross'] = ((ma50 < ma200) & (ma50.shift(1) >= ma200.shift(1))).astype(int)

        # 변동성
        features['volatility'] = df['close'].pct_change().rolling(window=20).std()
        features['high_low_ratio'] = (df['high'] - df['low']) / df['close']

        # 시간 특징
        features['hour'] = df.index.hour
        features['day_of_week'] = df.index.dayofweek

        return features.fillna(0)

    def get_probabilities(self, timeframe):
        """상승/하락 확률 계산"""
        if timeframe not in self.models:
            return None, None, None

        try:
            # 데이터 수집
            limit = 250
            ohlcv = self.exchange.fetch_ohlcv('BTC/USDT', timeframe, limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)

            combined_model = self.models[timeframe]
            up_model_info = combined_model['up_model']
            down_model_info = combined_model['down_model']

            # 상승 확률 계산
            up_features = self.create_specialized_features(df, 'up')
            up_features = up_features[up_model_info['features']].iloc[-1:].fillna(0)
            up_X_scaled = up_model_info['scaler'].transform(up_features)
            up_proba = up_model_info['model'].predict_proba(up_X_scaled)[0, 1]

            # 하락 확률 계산
            down_features = self.create_specialized_features(df, 'down')
            down_features = down_features[down_model_info['features']].iloc[-1:].fillna(0)
            down_X_scaled = down_model_info['scaler'].transform(down_features)
            down_proba = down_model_info['model'].predict_proba(down_X_scaled)[0, 1]

            # 최종 신호 결정
            if up_proba > 0.6 and up_proba > down_proba:
                signal = "UP"
            elif down_proba > 0.6 and down_proba > up_proba:
                signal = "DOWN"
            else:
                signal = "NEUTRAL"

            return up_proba, down_proba, signal

        except Exception as e:
            print(f"예측 오류 ({timeframe}): {e}")
            return None, None, None

    def get_all_probabilities(self):
        """모든 타임프레임 확률 조회"""
        results = {}

        for timeframe in ['15m', '30m', '1h', '4h']:
            up_prob, down_prob, signal = self.get_probabilities(timeframe)

            if up_prob is not None:
                results[timeframe] = {
                    'up_probability': up_prob,
                    'down_probability': down_prob,
                    'signal': signal,
                    'timestamp': datetime.now().isoformat()
                }

        return results

if __name__ == "__main__":
    predictor = SpecialistPredictor()

    print("\n" + "=" * 60)
    print("🔮 전문 모델 확률 예측")
    print("=" * 60)

    results = predictor.get_all_probabilities()

    for timeframe, data in results.items():
        print(f"\n{timeframe}:")
        print(f"  📈 상승 확률: {data['up_probability']:.1%}")
        print(f"  📉 하락 확률: {data['down_probability']:.1%}")
        print(f"  🎯 신호: {data['signal']}")