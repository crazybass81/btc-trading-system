#!/usr/bin/env python3
"""
Fast ML Models for BTC Direction Prediction
최적화된 빠른 훈련 버전
"""

import ccxt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import joblib
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class FastMLTrainer:
    def __init__(self):
        self.exchange = ccxt.binance()

    def get_data(self, timeframe, limit=2000):
        """데이터 수집 (빠른 버전)"""
        print(f"📊 {timeframe} 데이터 수집 ({limit}개 캔들)...")

        ohlcv = self.exchange.fetch_ohlcv('BTC/USDT', timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)

        print(f"  ✅ {len(df)}개 캔들 수집 완료")
        return df

    def create_features(self, df, direction='up'):
        """핵심 특징만 생성 (빠른 버전)"""
        features = pd.DataFrame(index=df.index)

        # 1. 수익률
        for period in [1, 3, 5, 10, 20]:
            features[f'return_{period}'] = df['close'].pct_change(period)

        # 2. RSI
        for period in [14, 21]:
            delta = df['close'].diff()
            gain = delta.where(delta > 0, 0).rolling(period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
            features[f'rsi_{period}'] = 100 - (100 / (1 + gain / (loss + 1e-10)))

        # 3. 이동평균
        for period in [10, 20, 50]:
            ma = df['close'].rolling(window=period).mean()
            features[f'ma_{period}_ratio'] = (df['close'] - ma) / (ma + 1e-10)

        # 4. 볼린저 밴드
        period = 20
        ma = df['close'].rolling(window=period).mean()
        std = df['close'].rolling(window=period).std()
        features['bb_upper'] = (df['close'] - (ma + 2*std)) / df['close']
        features['bb_lower'] = ((ma - 2*std) - df['close']) / df['close']
        features['bb_width'] = (4*std) / (ma + 1e-10)

        # 5. MACD
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9, adjust=False).mean()
        features['macd'] = macd / (df['close'] + 1e-10)
        features['macd_signal'] = signal / (df['close'] + 1e-10)
        features['macd_hist'] = (macd - signal) / (df['close'] + 1e-10)

        # 6. 볼륨
        features['volume_ratio'] = df['volume'] / df['volume'].rolling(window=20).mean()
        features['volume_change'] = df['volume'].pct_change()

        # 7. 변동성
        features['volatility'] = df['close'].pct_change().rolling(window=20).std()
        features['high_low_ratio'] = (df['high'] - df['low']) / df['close']

        # 8. 가격 위치
        highest_20 = df['high'].rolling(window=20).max()
        lowest_20 = df['low'].rolling(window=20).min()
        features['price_position'] = (df['close'] - lowest_20) / (highest_20 - lowest_20 + 1e-10)

        # 9. 트렌드
        features['sma_trend'] = (df['close'].rolling(10).mean() - df['close'].rolling(50).mean()) / df['close']

        # 10. 방향별 특화 특징
        if direction == 'up':
            features['bullish_candle'] = ((df['close'] > df['open']) * 1.0).rolling(5).mean()
            features['higher_highs'] = (df['high'] > df['high'].shift(1)).rolling(5).sum()
        else:
            features['bearish_candle'] = ((df['close'] < df['open']) * 1.0).rolling(5).mean()
            features['lower_lows'] = (df['low'] < df['low'].shift(1)).rolling(5).sum()

        # 시간 특징
        features['hour'] = df.index.hour
        features['day_of_week'] = df.index.dayofweek

        # NaN 처리
        features = features.replace([np.inf, -np.inf], np.nan)
        features = features.fillna(0)

        return features

    def create_labels(self, df, direction='up', timeframe='15m'):
        """라벨 생성"""
        # 예측 horizon (캔들 수)
        horizons = {
            '15m': 8,   # 2시간
            '30m': 4,   # 2시간
            '1h': 2,    # 2시간
        }

        horizon = horizons.get(timeframe, 4)

        # 미래 수익률
        future_return = df['close'].shift(-horizon) / df['close'] - 1

        # 임계값
        thresholds = {
            '15m': 0.002,  # 0.2%
            '30m': 0.003,  # 0.3%
            '1h': 0.004,   # 0.4%
        }

        threshold = thresholds.get(timeframe, 0.003)

        if direction == 'up':
            labels = (future_return > threshold).astype(int)
        else:
            labels = (future_return < -threshold).astype(int)

        return labels

    def train_model(self, timeframe, direction='up'):
        """모델 훈련"""
        print(f"\n{'='*60}")
        print(f"🚀 {timeframe} {direction.upper()} 모델 훈련")
        print(f"{'='*60}")

        # 데이터 수집
        df = self.get_data(timeframe, limit=2000)

        # 특징 및 라벨 생성
        print(f"  📐 특징 생성 중...")
        features = self.create_features(df, direction)
        labels = self.create_labels(df, direction, timeframe)

        # 유효 데이터
        valid_idx = ~(features.isna().any(axis=1) | labels.isna())
        X = features[valid_idx]
        y = labels[valid_idx]

        print(f"  📊 데이터: {len(X)}개 샘플, {X.shape[1]}개 특징")
        print(f"  📈 타겟 비율: {y.mean():.1%}")

        # 데이터 분할
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=False
        )

        # 스케일링
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # 모델 훈련
        models = {}

        # 1. XGBoost
        print(f"  🔧 XGBoost 훈련 중...")
        xgb = XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss'
        )
        xgb.fit(X_train_scaled, y_train)
        xgb_pred = xgb.predict(X_test_scaled)
        xgb_acc = accuracy_score(y_test, xgb_pred)
        xgb_prec = precision_score(y_test, xgb_pred, zero_division=0)
        xgb_rec = recall_score(y_test, xgb_pred, zero_division=0)
        xgb_f1 = f1_score(y_test, xgb_pred, zero_division=0)

        models['xgboost'] = {
            'model': xgb,
            'accuracy': xgb_acc,
            'precision': xgb_prec,
            'recall': xgb_rec,
            'f1': xgb_f1
        }

        print(f"    정확도: {xgb_acc:.1%}, 정밀도: {xgb_prec:.1%}, 재현율: {xgb_rec:.1%}")

        # 2. LightGBM
        print(f"  🔧 LightGBM 훈련 중...")
        lgb = LGBMClassifier(
            n_estimators=300,
            num_leaves=31,
            max_depth=5,
            learning_rate=0.05,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbosity=-1
        )
        lgb.fit(X_train_scaled, y_train)
        lgb_pred = lgb.predict(X_test_scaled)
        lgb_acc = accuracy_score(y_test, lgb_pred)
        lgb_prec = precision_score(y_test, lgb_pred, zero_division=0)
        lgb_rec = recall_score(y_test, lgb_pred, zero_division=0)
        lgb_f1 = f1_score(y_test, lgb_pred, zero_division=0)

        models['lightgbm'] = {
            'model': lgb,
            'accuracy': lgb_acc,
            'precision': lgb_prec,
            'recall': lgb_rec,
            'f1': lgb_f1
        }

        print(f"    정확도: {lgb_acc:.1%}, 정밀도: {lgb_prec:.1%}, 재현율: {lgb_rec:.1%}")

        # 앙상블
        print(f"  🎯 앙상블 예측...")
        ensemble_pred = (xgb_pred + lgb_pred) / 2 > 0.5
        ensemble_acc = accuracy_score(y_test, ensemble_pred)
        ensemble_prec = precision_score(y_test, ensemble_pred, zero_division=0)
        ensemble_rec = recall_score(y_test, ensemble_pred, zero_division=0)
        ensemble_f1 = f1_score(y_test, ensemble_pred, zero_division=0)

        print(f"    앙상블 정확도: {ensemble_acc:.1%}, F1: {ensemble_f1:.3f}")

        # 최고 모델 선택
        best_model_name = max(models, key=lambda x: models[x]['accuracy'])
        best_model = models[best_model_name]

        print(f"\n  🏆 최고 모델: {best_model_name}")
        print(f"     정확도: {best_model['accuracy']:.1%}")
        print(f"     정밀도: {best_model['precision']:.1%}")
        print(f"     재현율: {best_model['recall']:.1%}")
        print(f"     F1 점수: {best_model['f1']:.3f}")

        # 모델 저장
        model_info = {
            'models': models,
            'scaler': scaler,
            'features': list(features.columns),
            'direction': direction,
            'timeframe': timeframe,
            'best_model': best_model_name,
            'best_accuracy': best_model['accuracy'],
            'ensemble_accuracy': ensemble_acc,
            'trained_at': datetime.now().isoformat()
        }

        filename = f"ml_{timeframe}_{direction}_model.pkl"
        joblib.dump(model_info, f"models/{filename}")
        print(f"\n  ✅ 모델 저장: models/{filename}")

        return model_info

def main():
    print("="*60)
    print("🚀 Fast ML 모델 훈련")
    print("📅 " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("="*60)

    trainer = FastMLTrainer()
    results = {}

    for timeframe in ['15m', '30m', '1h']:
        for direction in ['up', 'down']:
            try:
                model_info = trainer.train_model(timeframe, direction)
                results[f"{timeframe}_{direction}"] = model_info
            except Exception as e:
                print(f"❌ {timeframe} {direction} 훈련 실패: {e}")

    # 결과 요약
    print("\n" + "="*60)
    print("📊 훈련 결과 요약")
    print("="*60)

    for key, info in results.items():
        tf, direction = key.rsplit('_', 1)
        print(f"\n{tf} {direction.upper()}:")
        print(f"  최고 모델: {info['best_model']}")
        print(f"  정확도: {info['best_accuracy']*100:.1f}%")
        print(f"  앙상블: {info['ensemble_accuracy']*100:.1f}%")

if __name__ == "__main__":
    main()