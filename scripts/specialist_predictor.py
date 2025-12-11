#!/usr/bin/env python3
"""
전문화 모델 예측 시스템
각 타임프레임별 상승/하락 확률을 독립적으로 예측
"""

import ccxt
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class SpecialistPredictor:
    def __init__(self):
        self.exchange = ccxt.binance()
        self.models = {}
        self.load_models()

    def load_models(self):
        """모든 전문 모델 로드"""
        timeframes = ['15m', '30m', '1h', '4h']

        for tf in timeframes:
            try:
                model_path = f"models/specialist_{tf}_combined_model.pkl"
                self.models[tf] = joblib.load(model_path)
                print(f"✅ {tf} 모델 로드 완료")
            except:
                print(f"⚠️ {tf} 모델 로드 실패")

    def get_current_data(self, timeframe, limit=100):
        """현재 데이터 수집"""
        ohlcv = self.exchange.fetch_ohlcv('BTC/USDT', timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        return df

    def create_features(self, df, direction='up'):
        """특징 생성"""
        features = pd.DataFrame(index=df.index)

        # 기본 특징
        for period in [1, 3, 5, 10, 20]:
            features[f'return_{period}'] = df['close'].pct_change(period)

        # 이동평균
        for period in [10, 20, 50, 100]:
            ma = df['close'].rolling(window=period).mean()
            features[f'ma_{period}_ratio'] = (df['close'] - ma) / ma

        # RSI
        for period in [7, 14, 21]:
            delta = df['close'].diff()
            gain = delta.where(delta > 0, 0).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / (loss + 1e-10)
            features[f'rsi_{period}'] = 100 - (100 / (1 + rs))

        # 볼린저 밴드
        for period in [20, 50]:
            ma = df['close'].rolling(window=period).mean()
            std = df['close'].rolling(window=period).std()
            features[f'bb_{period}_upper'] = (df['close'] - (ma + 2*std)) / df['close']
            features[f'bb_{period}_lower'] = ((ma - 2*std) - df['close']) / df['close']
            features[f'bb_{period}_width'] = (2*std) / ma

        # MACD
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9, adjust=False).mean()
        features['macd'] = macd / df['close']
        features['macd_signal'] = signal / df['close']
        features['macd_hist'] = (macd - signal) / df['close']

        # 볼륨
        features['volume_ratio'] = df['volume'] / df['volume'].rolling(window=20).mean()
        features['volume_change'] = df['volume'].pct_change()

        # 변동성
        features['volatility'] = df['close'].rolling(window=20).std() / df['close'].rolling(window=20).mean()
        features['high_low_ratio'] = (df['high'] - df['low']) / df['close']

        # 방향별 특화 특징
        if direction == 'up':
            features['up_volume'] = (df['close'] > df['open']).astype(int) * df['volume']
            features['up_momentum'] = (df['close'].pct_change(5) > 0).astype(int)
            features['support_break'] = (df['low'] < df['low'].rolling(20).min()).astype(int)
            features['bullish_ratio'] = (df['close'] > df['open']).rolling(10).sum() / 10
            features['new_high_20'] = (df['high'] == df['high'].rolling(20).max()).astype(int)
            features['new_high_50'] = (df['high'] == df['high'].rolling(50).max()).astype(int)
        else:
            features['down_volume'] = (df['close'] < df['open']).astype(int) * df['volume']
            features['down_momentum'] = (df['close'].pct_change(5) < 0).astype(int)
            features['resistance_break'] = (df['high'] > df['high'].rolling(20).max()).astype(int)
            features['bearish_ratio'] = (df['close'] < df['open']).rolling(10).sum() / 10
            features['new_low_20'] = (df['low'] == df['low'].rolling(20).min()).astype(int)
            features['new_low_50'] = (df['low'] == df['low'].rolling(50).min()).astype(int)

        # 시간 특징
        features['hour'] = df.index.hour
        features['day_of_week'] = df.index.dayofweek

        # NaN 처리
        features = features.replace([np.inf, -np.inf], np.nan)
        features = features.fillna(0)

        return features

    def predict_direction(self, timeframe):
        """특정 타임프레임의 상승/하락 확률 예측"""
        if timeframe not in self.models:
            return None

        # 데이터 수집
        df = self.get_current_data(timeframe)

        # 모델 정보
        model_info = self.models[timeframe]

        predictions = {}

        # 상승 예측
        if 'up_model' in model_info:
            up_features = self.create_features(df, 'up')
            up_model = model_info['up_model']
            up_scaler = model_info['up_scaler']

            # 최신 데이터만
            X_up = up_features.iloc[-1:][model_info['up_features']]
            X_up_scaled = up_scaler.transform(X_up)

            up_proba = up_model.predict_proba(X_up_scaled)[0, 1]
            predictions['up_probability'] = up_proba

        # 하락 예측
        if 'down_model' in model_info:
            down_features = self.create_features(df, 'down')
            down_model = model_info['down_model']
            down_scaler = model_info['down_scaler']

            # 최신 데이터만
            X_down = down_features.iloc[-1:][model_info['down_features']]
            X_down_scaled = down_scaler.transform(X_down)

            down_proba = down_model.predict_proba(X_down_scaled)[0, 1]
            predictions['down_probability'] = down_proba

        # 현재 가격 정보
        predictions['current_price'] = df['close'].iloc[-1]
        predictions['timestamp'] = df.index[-1]

        return predictions

    def get_all_predictions(self):
        """모든 타임프레임 예측"""
        all_predictions = {}

        for timeframe in self.models.keys():
            try:
                pred = self.predict_direction(timeframe)
                if pred:
                    all_predictions[timeframe] = pred
            except Exception as e:
                print(f"❌ {timeframe} 예측 실패: {e}")

        return all_predictions

    def get_trading_signal(self):
        """종합 거래 신호 생성"""
        predictions = self.get_all_predictions()

        if not predictions:
            return None

        # 가중 평균 계산
        weights = {'15m': 1.0, '30m': 2.0, '1h': 1.5, '4h': 1.0}

        weighted_up = 0
        weighted_down = 0
        total_weight = 0

        for tf, pred in predictions.items():
            weight = weights.get(tf, 1.0)

            if 'up_probability' in pred:
                weighted_up += pred['up_probability'] * weight

            if 'down_probability' in pred:
                weighted_down += pred['down_probability'] * weight

            total_weight += weight

        if total_weight > 0:
            avg_up = weighted_up / total_weight
            avg_down = weighted_down / total_weight

            # 신호 결정
            signal = {
                'timestamp': datetime.now(),
                'up_probability': avg_up,
                'down_probability': avg_down,
                'signal': 'NEUTRAL',
                'confidence': 0
            }

            # 강한 신호 기준
            if avg_up > 0.60 and avg_up > avg_down * 1.5:
                signal['signal'] = 'STRONG_BUY'
                signal['confidence'] = avg_up
            elif avg_up > 0.55 and avg_up > avg_down * 1.2:
                signal['signal'] = 'BUY'
                signal['confidence'] = avg_up
            elif avg_down > 0.60 and avg_down > avg_up * 1.5:
                signal['signal'] = 'STRONG_SELL'
                signal['confidence'] = avg_down
            elif avg_down > 0.55 and avg_down > avg_up * 1.2:
                signal['signal'] = 'SELL'
                signal['confidence'] = avg_down

            # 개별 타임프레임 예측 추가
            signal['timeframes'] = predictions

            return signal

        return None

def main():
    predictor = SpecialistPredictor()

    print("\n" + "="*60)
    print("🔮 전문화 모델 예측 시스템")
    print("="*60)

    # 모든 타임프레임 예측
    predictions = predictor.get_all_predictions()

    print("\n📊 개별 타임프레임 예측:")
    print("-"*40)

    for tf, pred in predictions.items():
        print(f"\n{tf}:")
        if 'up_probability' in pred:
            print(f"  📈 상승 확률: {pred['up_probability']*100:.1f}%")
        if 'down_probability' in pred:
            print(f"  📉 하락 확률: {pred['down_probability']*100:.1f}%")

    # 종합 신호
    signal = predictor.get_trading_signal()

    if signal:
        print("\n" + "="*60)
        print("🎯 종합 거래 신호")
        print("="*60)
        print(f"\n신호: {signal['signal']}")
        print(f"신뢰도: {signal['confidence']*100:.1f}%")
        print(f"상승 확률 (가중평균): {signal['up_probability']*100:.1f}%")
        print(f"하락 확률 (가중평균): {signal['down_probability']*100:.1f}%")

        # 거래 추천
        print("\n💡 거래 추천:")
        if signal['signal'] == 'STRONG_BUY':
            print("  ✅ 강력 매수 - 즉시 포지션 진입 추천")
        elif signal['signal'] == 'BUY':
            print("  ✅ 매수 - 분할 매수 추천")
        elif signal['signal'] == 'STRONG_SELL':
            print("  ❌ 강력 매도 - 즉시 포지션 청산 추천")
        elif signal['signal'] == 'SELL':
            print("  ❌ 매도 - 분할 매도 추천")
        else:
            print("  ⏸️ 대기 - 명확한 신호 없음")

if __name__ == "__main__":
    main()