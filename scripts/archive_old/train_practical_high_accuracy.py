#!/usr/bin/env python3
"""
실용적인 고정확도 모델 훈련
- GridSearch 없이 검증된 하이퍼파라미터 사용
- 빠른 훈련 시간
- 실전 사용 가능
"""

import ccxt
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import joblib
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class PracticalHighAccuracyTrainer:
    def __init__(self):
        self.exchange = ccxt.binance()

    def get_data(self, timeframe, limit=10000):
        """데이터 수집"""
        print(f"📊 데이터 수집: {timeframe}")

        all_data = []
        chunk_size = 1000

        for i in range(limit // chunk_size):
            try:
                since = None
                if all_data:
                    since = all_data[-1][0] - (chunk_size * 60000)

                ohlcv = self.exchange.fetch_ohlcv(
                    'BTC/USDT', timeframe, limit=chunk_size, since=since
                )

                if all_data:
                    ohlcv = [x for x in ohlcv if x[0] < all_data[0][0]]

                all_data = ohlcv + all_data

                if len(all_data) >= limit:
                    break

            except Exception as e:
                print(f"  ⚠️ 데이터 수집 중단: {e}")
                break

        df = pd.DataFrame(all_data[:limit], columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)

        print(f"  ✅ {len(df)}개 캔들 수집 완료")
        return df

    def create_practical_features(self, df):
        """실용적인 핵심 특징만 생성"""
        features = pd.DataFrame(index=df.index)

        # 1. 가격 변화율 (핵심)
        for period in [1, 3, 5, 10, 20, 50]:
            features[f'return_{period}'] = df['close'].pct_change(period)

        # 2. 이동평균
        for period in [10, 20, 50, 100]:
            ma = df['close'].rolling(window=period).mean()
            features[f'ma_{period}_ratio'] = (df['close'] - ma) / ma
            features[f'ma_{period}_slope'] = ma.pct_change(5)

        # 3. RSI
        for period in [14, 21]:
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / (loss + 1e-10)
            features[f'rsi_{period}'] = 100 - (100 / (1 + rs))

        # 4. MACD
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9, adjust=False).mean()
        features['macd'] = macd / df['close']
        features['macd_signal'] = signal / df['close']
        features['macd_hist'] = (macd - signal) / df['close']

        # 5. 볼린저 밴드
        for period in [20]:
            ma = df['close'].rolling(window=period).mean()
            std = df['close'].rolling(window=period).std()
            features[f'bb_{period}_position'] = (df['close'] - ma) / (2 * std)
            features[f'bb_{period}_width'] = (2 * std) / ma

        # 6. ATR (변동성)
        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift())
        low_close = abs(df['low'] - df['close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        features['atr_14'] = tr.rolling(window=14).mean() / df['close']

        # 7. 볼륨
        features['volume_ratio'] = df['volume'] / df['volume'].rolling(window=20).mean()
        features['volume_ma_ratio'] = df['volume'].rolling(window=5).mean() / df['volume'].rolling(window=20).mean()

        # 8. 지지/저항
        features['dist_from_high_20'] = (df['high'].rolling(window=20).max() - df['close']) / df['close']
        features['dist_from_low_20'] = (df['close'] - df['low'].rolling(window=20).min()) / df['close']

        # 9. 캔들 패턴
        body = df['close'] - df['open']
        features['body_ratio'] = body / (df['high'] - df['low'] + 1e-10)
        features['upper_shadow'] = (df['high'] - df[['close', 'open']].max(axis=1)) / (df['high'] - df['low'] + 1e-10)
        features['lower_shadow'] = (df[['close', 'open']].min(axis=1) - df['low']) / (df['high'] - df['low'] + 1e-10)

        # 10. 시간 특징
        features['hour'] = df.index.hour
        features['day_of_week'] = df.index.dayofweek

        return features.fillna(0)

    def create_smart_labels(self, df, timeframe):
        """스마트 라벨 생성"""
        thresholds = {
            '15m': 0.0015,
            '30m': 0.002,
            '1h': 0.0025,
            '4h': 0.003
        }

        threshold = thresholds.get(timeframe, 0.002)

        # 미래 수익률 (복합)
        returns = []
        weights = [0.5, 0.3, 0.2]

        for i, w in enumerate(weights, 1):
            ret = (df['close'].shift(-i) / df['close'] - 1)
            returns.append(ret * w)

        weighted_return = sum(returns)

        # 라벨 생성
        labels = pd.Series(index=df.index, dtype=int)
        labels[weighted_return > threshold] = 1  # UP
        labels[weighted_return < -threshold] = 0  # DOWN

        # 애매한 경우 제거
        labels = labels.dropna()

        return labels

    def train_optimized_models(self, timeframe, direction='both'):
        """최적화된 모델 훈련"""
        print(f"\n{'='*60}")
        print(f"🚀 {timeframe} 실용적 고정확도 모델 훈련")
        print(f"{'='*60}")

        # 데이터 수집
        df = self.get_data(timeframe, limit=10000)

        # 특징 생성
        print("  📐 특징 생성 중...")
        features = self.create_practical_features(df)

        # 라벨 생성
        labels = self.create_smart_labels(df, timeframe)

        # 유효 데이터
        valid_idx = features.index.intersection(labels.index)
        X = features.loc[valid_idx]
        y = labels.loc[valid_idx]

        # NaN 제거
        valid_mask = ~(X.isna().any(axis=1) | y.isna())
        X = X[valid_mask]
        y = y[valid_mask]

        print(f"  📊 데이터: {len(X)}개 샘플")
        print(f"  📈 UP: {(y==1).sum()}개 ({(y==1).sum()/len(y)*100:.1f}%)")
        print(f"  📉 DOWN: {(y==0).sum()}개 ({(y==0).sum()/len(y)*100:.1f}%)")

        if len(X) < 100:
            print("  ⚠️ 데이터 부족")
            return None

        # 데이터 분할
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        # 스케일링
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # 모델 정의 (검증된 하이퍼파라미터)
        models = {
            'xgboost': XGBClassifier(
                n_estimators=300,
                max_depth=7,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                use_label_encoder=False,
                eval_metric='logloss'
            ),
            'lightgbm': LGBMClassifier(
                n_estimators=300,
                max_depth=7,
                learning_rate=0.05,
                num_leaves=50,
                min_child_samples=20,
                random_state=42,
                verbosity=-1
            ),
            'rf': RandomForestClassifier(
                n_estimators=300,
                max_depth=15,
                min_samples_split=10,
                min_samples_leaf=5,
                max_features='sqrt',
                random_state=42,
                n_jobs=-1
            ),
            'gb': GradientBoostingClassifier(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.05,
                min_samples_split=20,
                min_samples_leaf=10,
                subsample=0.8,
                random_state=42
            )
        }

        # 모델 훈련 및 평가
        print("\n  📊 모델 훈련 및 평가:")
        results = {}

        for name, model in models.items():
            print(f"    {name}...", end=' ')

            # 훈련
            model.fit(X_train_scaled, y_train)

            # 예측
            y_pred = model.predict(X_test_scaled)
            y_proba = model.predict_proba(X_test_scaled)

            # 평가
            acc = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)

            results[name] = {
                'model': model,
                'accuracy': acc,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'predictions': y_pred,
                'probabilities': y_proba
            }

            print(f"Acc={acc:.3f}, F1={f1:.3f}")

        # 앙상블 예측
        print("\n  🎯 앙상블 예측:")

        # 소프트 보팅
        ensemble_proba = np.zeros((len(X_test_scaled), 2))
        weights = {'xgboost': 1.5, 'lightgbm': 1.5, 'rf': 1.0, 'gb': 1.2}

        for name, result in results.items():
            weight = weights.get(name, 1.0)
            ensemble_proba += result['probabilities'] * weight

        ensemble_proba /= sum(weights.values())
        ensemble_pred = (ensemble_proba[:, 1] > 0.5).astype(int)

        # 앙상블 평가
        ensemble_acc = accuracy_score(y_test, ensemble_pred)
        ensemble_precision = precision_score(y_test, ensemble_pred, zero_division=0)
        ensemble_recall = recall_score(y_test, ensemble_pred, zero_division=0)
        ensemble_f1 = f1_score(y_test, ensemble_pred, zero_division=0)

        print(f"    앙상블: Acc={ensemble_acc:.3f}, P={ensemble_precision:.3f}, R={ensemble_recall:.3f}, F1={ensemble_f1:.3f}")

        # 혼동 행렬
        cm = confusion_matrix(y_test, ensemble_pred)
        print(f"\n  혼동 행렬:")
        print(f"           예측DOWN  예측UP")
        print(f"  실제DOWN    {cm[0,0]:4d}    {cm[0,1]:4d}")
        print(f"  실제UP      {cm[1,0]:4d}    {cm[1,1]:4d}")

        # 각 클래스 정확도
        if cm[0,0] + cm[0,1] > 0:
            down_acc = cm[0,0] / (cm[0,0] + cm[0,1]) * 100
            print(f"\n  DOWN 예측 정확도: {down_acc:.1f}%")
        if cm[1,0] + cm[1,1] > 0:
            up_acc = cm[1,1] / (cm[1,0] + cm[1,1]) * 100
            print(f"  UP 예측 정확도: {up_acc:.1f}%")

        # 모델 저장
        if ensemble_acc > 0.55:  # 55% 이상만 저장
            model_info = {
                'models': {name: r['model'] for name, r in results.items()},
                'scaler': scaler,
                'features': list(features.columns),
                'ensemble_accuracy': ensemble_acc,
                'ensemble_precision': ensemble_precision,
                'ensemble_recall': ensemble_recall,
                'ensemble_f1': ensemble_f1,
                'individual_results': {
                    name: {
                        'accuracy': r['accuracy'],
                        'f1': r['f1']
                    } for name, r in results.items()
                },
                'timeframe': timeframe,
                'data_size': len(X),
                'trained_at': datetime.now().isoformat()
            }

            filename = f"practical_high_acc_{timeframe}_model.pkl"
            joblib.dump(model_info, f"models/{filename}")
            print(f"\n  ✅ 모델 저장: models/{filename}")

            return model_info
        else:
            print(f"\n  ⚠️ 정확도 부족 ({ensemble_acc:.1%})")
            return None

def main():
    trainer = PracticalHighAccuracyTrainer()

    print("=" * 60)
    print("🔧 실용적 고정확도 모델 훈련")
    print("⏰ 예상 시간: 타임프레임당 1-2분")
    print("=" * 60)

    results = {}

    for timeframe in ['15m', '30m', '1h', '4h']:
        try:
            model_info = trainer.train_optimized_models(timeframe)
            if model_info:
                results[timeframe] = model_info
        except Exception as e:
            print(f"\n❌ {timeframe} 훈련 실패: {e}")

    # 결과 요약
    if results:
        print("\n" + "=" * 60)
        print("📋 훈련 결과 요약")
        print("=" * 60)

        for tf, info in results.items():
            print(f"\n{tf}:")
            print(f"  앙상블 정확도: {info['ensemble_accuracy']*100:.1f}%")
            print(f"  F1 점수: {info['ensemble_f1']:.3f}")

            # 개별 모델 성능
            print(f"  개별 모델:")
            for name, res in info['individual_results'].items():
                print(f"    {name}: Acc={res['accuracy']:.3f}, F1={res['f1']:.3f}")

if __name__ == "__main__":
    main()