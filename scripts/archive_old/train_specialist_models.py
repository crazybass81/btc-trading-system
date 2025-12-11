#!/usr/bin/env python3
"""
전문화된 상승/하락 모델 훈련
- 상승 전문 모델: 상승 패턴만 학습
- 하락 전문 모델: 하락 패턴만 학습
- 결합 예측: 두 모델의 신뢰도 비교
"""

import ccxt
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from xgboost import XGBClassifier
import joblib
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class SpecialistModelTrainer:
    def __init__(self):
        self.exchange = ccxt.binance()

    def get_data(self, timeframe, limit=10000):
        """데이터 수집"""
        print(f"📊 데이터 수집: {timeframe} ({limit}개 캔들)")

        ohlcv = self.exchange.fetch_ohlcv('BTC/USDT', timeframe=timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)

        return df

    def create_specialized_features(self, df, direction='up'):
        """방향별 특화 특징 생성"""
        features = pd.DataFrame(index=df.index)

        # 공통 특징
        # 가격 변화율
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
            # 지지선 돌파
            features['support_break'] = df['close'] / df['low'].rolling(window=20).min() - 1

            # 상승 모멘텀
            features['up_momentum'] = (df['close'] > df['close'].shift(1)).rolling(window=10).sum()

            # 고점 갱신
            features['new_high_20'] = (df['high'] == df['high'].rolling(window=20).max()).astype(int)
            features['new_high_50'] = (df['high'] == df['high'].rolling(window=50).max()).astype(int)

            # 양봉 비율
            features['bullish_ratio'] = ((df['close'] > df['open']).rolling(window=10).sum()) / 10

            # 상승 거래량
            up_days = df['close'] > df['close'].shift(1)
            features['up_volume'] = (df['volume'] * up_days).rolling(window=10).sum()

            # 골든크로스
            ma50 = df['close'].rolling(window=50).mean()
            ma200 = df['close'].rolling(window=200).mean()
            features['golden_cross'] = ((ma50 > ma200) & (ma50.shift(1) <= ma200.shift(1))).astype(int)

        else:  # down
            # 하락 특화 특징
            # 저항선 돌파
            features['resistance_break'] = 1 - df['close'] / df['high'].rolling(window=20).max()

            # 하락 모멘텀
            features['down_momentum'] = (df['close'] < df['close'].shift(1)).rolling(window=10).sum()

            # 저점 갱신
            features['new_low_20'] = (df['low'] == df['low'].rolling(window=20).min()).astype(int)
            features['new_low_50'] = (df['low'] == df['low'].rolling(window=50).min()).astype(int)

            # 음봉 비율
            features['bearish_ratio'] = ((df['close'] < df['open']).rolling(window=10).sum()) / 10

            # 하락 거래량
            down_days = df['close'] < df['close'].shift(1)
            features['down_volume'] = (df['volume'] * down_days).rolling(window=10).sum()

            # 데드크로스
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

    def create_specialist_labels(self, df, direction='up', timeframe='15m'):
        """전문화된 라벨 생성"""
        # 타임프레임별 임계값
        thresholds = {
            '15m': 0.001,   # 0.1%
            '30m': 0.0015,  # 0.15%
            '1h': 0.002,    # 0.2%
            '4h': 0.003     # 0.3%
        }

        threshold = thresholds.get(timeframe, 0.002)

        # 미래 수익률
        future_return = df['close'].shift(-1) / df['close'] - 1

        if direction == 'up':
            # 상승 모델: 상승은 1, 나머지는 0
            labels = (future_return > threshold).astype(int)
        else:
            # 하락 모델: 하락은 1, 나머지는 0
            labels = (future_return < -threshold).astype(int)

        return labels

    def train_specialist_model(self, timeframe, direction='up'):
        """전문 모델 훈련"""
        print(f"\n{'='*60}")
        print(f"🚀 {timeframe} {direction.upper()} 전문 모델 훈련")
        print(f"{'='*60}")

        # 데이터 수집
        df = self.get_data(timeframe)

        # 특징 생성
        features = self.create_specialized_features(df, direction)

        # 라벨 생성
        labels = self.create_specialist_labels(df, direction, timeframe)

        # 유효 데이터
        valid_idx = ~(features.isna().any(axis=1) | labels.isna())
        X = features[valid_idx]
        y = labels[valid_idx]

        print(f"훈련 데이터: {len(X)}개")
        print(f"타겟 클래스 비율: {y.mean():.1%}")

        # 데이터 분할
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        # 스케일링
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # 모델 선택 (방향별 최적화)
        if direction == 'up':
            # 상승 전문: GradientBoosting (보수적)
            model = GradientBoostingClassifier(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=5,
                min_samples_split=20,
                min_samples_leaf=10,
                subsample=0.8,
                random_state=42
            )
        else:
            # 하락 전문: XGBoost (민감)
            model = XGBClassifier(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=5,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                use_label_encoder=False,
                eval_metric='logloss'
            )

        # 훈련
        model.fit(X_train_scaled, y_train)

        # 평가
        y_pred = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled)[:, 1]

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        try:
            auc = roc_auc_score(y_test, y_proba)
        except:
            auc = 0.5

        print(f"\n📊 모델 평가:")
        print(f"정확도: {accuracy*100:.1f}%")
        print(f"정밀도: {precision*100:.1f}%")
        print(f"재현율: {recall*100:.1f}%")
        print(f"F1 점수: {f1:.3f}")
        print(f"AUC: {auc:.3f}")

        # 특징 중요도 (상위 10개)
        if hasattr(model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': X.columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)

            print(f"\n🔑 중요 특징 (상위 10개):")
            for i, row in feature_importance.head(10).iterrows():
                print(f"  {row['feature']}: {row['importance']:.3f}")

        # 모델 정보 저장
        model_info = {
            'model': model,
            'scaler': scaler,
            'features': list(features.columns),
            'direction': direction,
            'timeframe': timeframe,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc': auc,
            'trained_at': datetime.now().isoformat()
        }

        return model_info

    def train_combined_specialist(self, timeframe):
        """상승/하락 전문 모델 결합"""
        print(f"\n{'='*60}")
        print(f"🎯 {timeframe} 결합 전문 모델")
        print(f"{'='*60}")

        # 상승 전문 모델 훈련
        up_model = self.train_specialist_model(timeframe, 'up')

        # 하락 전문 모델 훈련
        down_model = self.train_specialist_model(timeframe, 'down')

        # 결합 모델 저장
        combined_info = {
            'up_model': up_model,
            'down_model': down_model,
            'timeframe': timeframe,
            'trained_at': datetime.now().isoformat()
        }

        filename = f"specialist_{timeframe}_combined_model.pkl"
        joblib.dump(combined_info, f"models/{filename}")
        print(f"\n✅ 결합 모델 저장: models/{filename}")

        return combined_info

    def predict_with_specialists(self, combined_model, df):
        """전문 모델로 예측"""
        up_model_info = combined_model['up_model']
        down_model_info = combined_model['down_model']

        # 상승 특징 생성 및 예측
        up_features = self.create_specialized_features(df, 'up')
        up_features = up_features[up_model_info['features']].iloc[-1:].fillna(0)
        up_X_scaled = up_model_info['scaler'].transform(up_features)
        up_proba = up_model_info['model'].predict_proba(up_X_scaled)[0, 1]

        # 하락 특징 생성 및 예측
        down_features = self.create_specialized_features(df, 'down')
        down_features = down_features[down_model_info['features']].iloc[-1:].fillna(0)
        down_X_scaled = down_model_info['scaler'].transform(down_features)
        down_proba = down_model_info['model'].predict_proba(down_X_scaled)[0, 1]

        print(f"\n📊 전문 모델 예측:")
        print(f"상승 확률: {up_proba:.1%}")
        print(f"하락 확률: {down_proba:.1%}")

        # 최종 예측
        if up_proba > down_proba and up_proba > 0.5:
            prediction = "UP"
            confidence = up_proba
        elif down_proba > up_proba and down_proba > 0.5:
            prediction = "DOWN"
            confidence = down_proba
        else:
            prediction = "NEUTRAL"
            confidence = max(1 - up_proba, 1 - down_proba)

        return prediction, confidence

def test_specialist_models():
    """전문 모델 테스트"""
    trainer = SpecialistModelTrainer()
    exchange = ccxt.binance()

    print("=" * 60)
    print("🧪 전문 모델 테스트")
    print("=" * 60)

    # 15분 모델 테스트
    timeframe = '15m'

    # 모델 훈련
    combined_model = trainer.train_combined_specialist(timeframe)

    # 최신 데이터로 예측
    ohlcv = exchange.fetch_ohlcv('BTC/USDT', timeframe, limit=200)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)

    prediction, confidence = trainer.predict_with_specialists(combined_model, df)

    print(f"\n🎯 최종 예측: {prediction} (신뢰도: {confidence:.1%})")

    # 실제 가격 정보
    current_price = df['close'].iloc[-1]
    price_change = df['close'].pct_change().iloc[-1] * 100

    print(f"\n현재 가격: ${current_price:,.0f} ({price_change:+.2f}%)")

def main():
    trainer = SpecialistModelTrainer()

    print("=" * 60)
    print("🔧 전문화 모델 훈련 시작")
    print("=" * 60)

    results = {}
    for timeframe in ['15m', '30m', '1h', '4h']:
        combined_model = trainer.train_combined_specialist(timeframe)
        results[timeframe] = combined_model

    # 결과 요약
    print("\n" + "=" * 60)
    print("📋 훈련 결과 요약")
    print("=" * 60)

    for tf, model_info in results.items():
        up_acc = model_info['up_model']['accuracy']
        down_acc = model_info['down_model']['accuracy']
        up_f1 = model_info['up_model']['f1_score']
        down_f1 = model_info['down_model']['f1_score']

        print(f"\n{tf}:")
        print(f"  상승 모델: 정확도 {up_acc*100:.1f}%, F1 {up_f1:.3f}")
        print(f"  하락 모델: 정확도 {down_acc*100:.1f}%, F1 {down_f1:.3f}")

if __name__ == "__main__":
    main()