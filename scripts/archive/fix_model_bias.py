#!/usr/bin/env python3
"""
모델 편향 문제 해결 스크립트
- 균형잡힌 데이터로 재훈련
- 클래스 가중치 적용
- 다양한 시장 상황 포함
"""

import ccxt
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from xgboost import XGBClassifier
import joblib
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class BalancedModelTrainer:
    def __init__(self):
        self.exchange = ccxt.binance()

        # 균형잡힌 임계값 설정 (더 민감하게)
        self.thresholds = {
            '15m': 0.0003,  # 0.03% - 매우 민감
            '30m': 0.0005,  # 0.05% - 민감
            '1h': 0.001,    # 0.1% - 적당
            '4h': 0.002     # 0.2% - 표준
        }

        # 데이터 수집 기간 (더 긴 기간)
        self.data_limits = {
            '15m': 10000,  # 약 4일
            '30m': 8000,   # 약 1주일
            '1h': 6000,    # 약 8개월
            '4h': 3000     # 약 16개월
        }

    def get_extended_data(self, timeframe):
        """더 긴 기간의 데이터 수집"""
        limit = self.data_limits[timeframe]
        print(f"\n📊 확장 데이터 수집: {timeframe} ({limit}개 캔들)")

        try:
            ohlcv = self.exchange.fetch_ohlcv('BTC/USDT', timeframe=timeframe, limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)

            # 기간 정보
            start_date = df.index[0].strftime('%Y-%m-%d')
            end_date = df.index[-1].strftime('%Y-%m-%d')
            print(f"  기간: {start_date} ~ {end_date}")

            return df
        except Exception as e:
            print(f"  ⚠️ 데이터 수집 실패: {e}")
            return None

    def create_balanced_labels(self, df, timeframe):
        """균형잡힌 라벨 생성"""
        threshold = self.thresholds[timeframe]

        # 미래 수익률 계산
        future_return = df['close'].shift(-1) / df['close'] - 1

        # 동적 임계값 (변동성 기반)
        volatility = future_return.rolling(window=100).std()
        dynamic_threshold = threshold * (1 + volatility)

        # 이진 분류 라벨 (0: DOWN, 1: UP)
        labels = (future_return > 0).astype(int)  # 단순히 상승/하락으로 구분

        # 분포 확인
        up_count = labels.sum()
        down_count = len(labels) - up_count
        total = len(labels)

        print(f"\n📈 라벨 분포 ({timeframe}):")
        print(f"  UP:   {up_count:5d}개 ({up_count/total*100:5.1f}%)")
        print(f"  DOWN: {down_count:5d}개 ({down_count/total*100:5.1f}%)")

        # 불균형 비율
        imbalance_ratio = max(up_count, down_count) / min(up_count, down_count)
        print(f"  불균형 비율: {imbalance_ratio:.2f}:1")

        return labels

    def create_enhanced_features(self, df):
        """향상된 특징 생성"""
        features = pd.DataFrame(index=df.index)

        # 가격 변화율
        for period in [1, 3, 5, 10, 20, 50]:
            features[f'return_{period}'] = df['close'].pct_change(period)

        # 이동평균
        for period in [5, 10, 20, 50, 100]:
            ma = df['close'].rolling(window=period).mean()
            features[f'ma_{period}_ratio'] = df['close'] / ma - 1
            features[f'ma_{period}_slope'] = ma.pct_change(5)

        # RSI
        for period in [7, 14, 21]:
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            features[f'rsi_{period}'] = 100 - (100 / (1 + rs))

        # 볼린저 밴드
        for period in [20, 50]:
            ma = df['close'].rolling(window=period).mean()
            std = df['close'].rolling(window=period).std()
            features[f'bb_{period}_upper'] = (df['close'] - (ma + 2*std)) / df['close']
            features[f'bb_{period}_lower'] = ((ma - 2*std) - df['close']) / df['close']
            features[f'bb_{period}_width'] = 4 * std / ma

        # 거래량 지표
        features['volume_ratio'] = df['volume'] / df['volume'].rolling(window=20).mean()
        features['volume_trend'] = df['volume'].rolling(window=10).mean().pct_change(5)

        # MACD
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9, adjust=False).mean()
        features['macd'] = macd / df['close']
        features['macd_signal'] = signal / df['close']
        features['macd_hist'] = (macd - signal) / df['close']

        # 변동성
        features['volatility'] = df['close'].pct_change().rolling(window=20).std()
        features['high_low_ratio'] = (df['high'] - df['low']) / df['close']

        # 시간 특징
        features['hour'] = df.index.hour
        features['day_of_week'] = df.index.dayofweek

        return features

    def train_balanced_model(self, timeframe, model_type='gradientboost'):
        """균형잡힌 모델 훈련"""
        print(f"\n{'='*60}")
        print(f"🚀 {timeframe} 균형 모델 훈련 시작")
        print(f"{'='*60}")

        # 데이터 수집
        df = self.get_extended_data(timeframe)
        if df is None:
            return None

        # 라벨 생성
        labels = self.create_balanced_labels(df, timeframe)

        # 특징 생성
        features = self.create_enhanced_features(df)

        # NaN 제거
        valid_idx = ~(features.isna().any(axis=1) | labels.isna())
        X = features[valid_idx]
        y = labels[valid_idx]

        print(f"\n훈련 데이터: {len(X)}개 샘플")

        # 클래스 가중치 계산
        class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
        class_weight_dict = {0: class_weights[0], 1: class_weights[1]}
        print(f"클래스 가중치: DOWN={class_weights[0]:.2f}, UP={class_weights[1]:.2f}")

        # 데이터 분할 (시계열 유지)
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        # 스케일링
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # 모델 선택 및 훈련
        if model_type == 'gradientboost':
            model = GradientBoostingClassifier(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=5,
                min_samples_split=20,
                min_samples_leaf=10,
                subsample=0.8,
                random_state=42
            )
            # GradientBoosting은 sample_weight 사용
            sample_weights = np.array([class_weight_dict[int(label)] for label in y_train])
            model.fit(X_train_scaled, y_train, sample_weight=sample_weights)

        elif model_type == 'xgboost':
            model = XGBClassifier(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=5,
                scale_pos_weight=class_weights[1]/class_weights[0],  # UP/DOWN 비율
                random_state=42,
                use_label_encoder=False,
                eval_metric='logloss'
            )
            model.fit(X_train_scaled, y_train)

        elif model_type == 'neuralnet':
            model = MLPClassifier(
                hidden_layer_sizes=(100, 50, 25),
                activation='relu',
                learning_rate_init=0.001,
                max_iter=500,
                random_state=42
            )
            # MLPClassifier는 sample_weight 사용
            sample_weights = np.array([class_weight_dict[int(label)] for label in y_train])
            model.fit(X_train_scaled, y_train)  # 일부 모델은 sample_weight를 지원하지 않음

        else:  # randomforest
            model = RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                min_samples_split=20,
                min_samples_leaf=10,
                class_weight=class_weight_dict,
                random_state=42
            )
            model.fit(X_train_scaled, y_train)

        # 평가
        y_pred_train = model.predict(X_train_scaled)
        y_pred_test = model.predict(X_test_scaled)

        print(f"\n📊 모델 평가:")
        print(f"훈련 정확도: {accuracy_score(y_train, y_pred_train)*100:.1f}%")
        print(f"테스트 정확도: {accuracy_score(y_test, y_pred_test)*100:.1f}%")

        # 혼동 행렬
        cm = confusion_matrix(y_test, y_pred_test)
        print(f"\n혼동 행렬:")
        print(f"         예측DOWN  예측UP")
        print(f"실제DOWN    {cm[0,0]:4d}    {cm[0,1]:4d}")
        print(f"실제UP      {cm[1,0]:4d}    {cm[1,1]:4d}")

        # 각 클래스별 정확도
        if len(cm) >= 2:
            down_acc = cm[0,0] / (cm[0,0] + cm[0,1]) * 100 if (cm[0,0] + cm[0,1]) > 0 else 0
            up_acc = cm[1,1] / (cm[1,0] + cm[1,1]) * 100 if (cm[1,0] + cm[1,1]) > 0 else 0
            print(f"\nDOWN 예측 정확도: {down_acc:.1f}%")
            print(f"UP 예측 정확도: {up_acc:.1f}%")

        # 예측 분포 확인
        pred_dist = pd.Series(y_pred_test).value_counts()
        print(f"\n예측 분포:")
        for val in [0, 1]:
            count = pred_dist.get(val, 0)
            pct = count / len(y_pred_test) * 100
            label = "DOWN" if val == 0 else "UP"
            print(f"  {label}: {count}개 ({pct:.1f}%)")

        # 최근 예측 확률 확인
        recent_probs = model.predict_proba(X_test_scaled[-10:])
        print(f"\n최근 10개 예측 확률:")
        for i, (down_prob, up_prob) in enumerate(recent_probs):
            actual = "UP" if y_test.iloc[-10+i] == 1 else "DOWN"
            pred = "UP" if up_prob > 0.5 else "DOWN"
            print(f"  {i+1:2d}: DOWN={down_prob:.1%}, UP={up_prob:.1%} | 실제={actual}, 예측={pred}")

        # 모델 저장
        model_info = {
            'model': model,
            'scaler': scaler,
            'features': list(features.columns),
            'accuracy': accuracy_score(y_test, y_pred_test),
            'timeframe': timeframe,
            'threshold': self.thresholds[timeframe],
            'trained_at': datetime.now().isoformat()
        }

        # 파일명
        filename = f"balanced_{timeframe}_{model_type}_model.pkl"
        filepath = f"models/{filename}"
        joblib.dump(model_info, filepath)
        print(f"\n✅ 모델 저장: {filepath}")

        return model_info

    def test_realtime_predictions(self, model_info):
        """실시간 예측 테스트"""
        timeframe = model_info['timeframe']
        print(f"\n🔮 실시간 예측 테스트 ({timeframe})")

        # 최신 데이터 가져오기
        ohlcv = self.exchange.fetch_ohlcv('BTC/USDT', timeframe=timeframe, limit=200)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)

        # 특징 생성
        features = self.create_enhanced_features(df)

        # 최신 데이터만 사용
        latest_features = features.iloc[-1:].dropna(axis=1, how='any')

        # 모델에 맞는 특징 선택
        model_features = [f for f in model_info['features'] if f in latest_features.columns]
        X = latest_features[model_features]

        if len(X) > 0 and len(model_features) == len(model_info['features']):
            # 예측
            X_scaled = model_info['scaler'].transform(X)
            prediction = model_info['model'].predict(X_scaled)[0]
            proba = model_info['model'].predict_proba(X_scaled)[0]

            # 현재 가격
            current_price = df['close'].iloc[-1]
            price_change = df['close'].pct_change().iloc[-1] * 100

            print(f"현재 가격: ${current_price:,.0f} ({price_change:+.2f}%)")
            print(f"예측: {'UP ↑' if prediction == 1 else 'DOWN ↓'}")
            print(f"확률: DOWN={proba[0]:.1%}, UP={proba[1]:.1%}")
        else:
            print("⚠️ 특징 생성 실패")

def main():
    trainer = BalancedModelTrainer()

    print("=" * 60)
    print("🔧 모델 편향 문제 해결 시작")
    print("=" * 60)

    # 각 타임프레임별로 균형잡힌 모델 훈련
    timeframes = ['15m', '30m', '1h', '4h']
    model_types = {
        '15m': 'gradientboost',
        '30m': 'neuralnet',
        '1h': 'gradientboost',
        '4h': 'neuralnet'
    }

    trained_models = {}

    for tf in timeframes:
        model_type = model_types[tf]
        model_info = trainer.train_balanced_model(tf, model_type)
        if model_info:
            trained_models[tf] = model_info
            trainer.test_realtime_predictions(model_info)

    # 결과 요약
    print("\n" + "=" * 60)
    print("📋 훈련 결과 요약")
    print("=" * 60)

    for tf, model_info in trained_models.items():
        print(f"\n{tf}: 정확도 {model_info['accuracy']*100:.1f}%")

        # 실제 예측 테스트
        ohlcv = trainer.exchange.fetch_ohlcv('BTC/USDT', timeframe=tf, limit=100)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)

        features = trainer.create_enhanced_features(df).iloc[-10:]
        valid_features = features.dropna()

        if len(valid_features) > 0:
            model_features = [f for f in model_info['features'] if f in valid_features.columns]
            if len(model_features) == len(model_info['features']):
                X = valid_features[model_features]
                X_scaled = model_info['scaler'].transform(X)
                predictions = model_info['model'].predict(X_scaled)

                up_count = (predictions == 1).sum()
                down_count = (predictions == 0).sum()
                print(f"  최근 {len(predictions)}개 예측: UP={up_count}, DOWN={down_count}")

if __name__ == "__main__":
    main()