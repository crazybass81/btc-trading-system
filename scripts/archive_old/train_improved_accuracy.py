#!/usr/bin/env python3
"""
개선된 고정확도 모델 훈련
- 더 많은 데이터 수집
- 데이터 품질 개선
- 특징 엔지니어링 최적화
"""

import ccxt
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import joblib
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class ImprovedAccuracyTrainer:
    def __init__(self):
        self.exchange = ccxt.binance()

    def get_extended_data(self, timeframe, days=90):
        """더 많은 데이터 수집 (90일)"""
        print(f"📊 확장 데이터 수집: {timeframe} ({days}일)")

        all_data = []
        chunk_size = 1000

        # 현재 시간부터 과거로
        end_time = self.exchange.milliseconds()

        # 타임프레임별 밀리초
        tf_ms = {
            '15m': 15 * 60 * 1000,
            '30m': 30 * 60 * 1000,
            '1h': 60 * 60 * 1000,
            '4h': 240 * 60 * 1000
        }

        ms_per_candle = tf_ms.get(timeframe, 60 * 60 * 1000)
        total_candles_needed = int(days * 24 * 60 * 60 * 1000 / ms_per_candle)

        collected = 0
        current_time = end_time

        while collected < total_candles_needed:
            try:
                # 과거 데이터 가져오기
                ohlcv = self.exchange.fetch_ohlcv(
                    'BTC/USDT',
                    timeframe,
                    limit=chunk_size,
                    since=current_time - (chunk_size * ms_per_candle)
                )

                if not ohlcv:
                    break

                all_data = ohlcv + all_data
                collected += len(ohlcv)

                # 다음 청크를 위해 시간 이동
                if ohlcv:
                    current_time = ohlcv[0][0]

                print(f"  수집 진행: {collected}/{total_candles_needed} ({collected/total_candles_needed*100:.1f}%)")

                if len(all_data) >= total_candles_needed:
                    all_data = all_data[-total_candles_needed:]
                    break

            except Exception as e:
                print(f"  ⚠️ 수집 중단: {e}")
                break

        df = pd.DataFrame(all_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)

        print(f"  ✅ {len(df)}개 캔들 수집 완료")
        return df

    def create_robust_features(self, df):
        """개선된 특징 생성 (infinity 방지)"""
        features = pd.DataFrame(index=df.index)

        # 안전한 epsilon 값
        eps = 1e-10

        # 1. 가격 변화율 (안전하게)
        for period in [1, 2, 3, 5, 8, 13, 21]:
            returns = df['close'].pct_change(period)
            features[f'return_{period}'] = returns.clip(-1, 1)  # 극단값 제한

        # 2. 이동평균 (안전한 계산)
        for period in [7, 14, 21, 50]:
            ma = df['close'].rolling(window=period, min_periods=1).mean()
            features[f'ma_{period}_ratio'] = ((df['close'] - ma) / (ma + eps)).clip(-1, 1)

        # 3. RSI (개선된 버전)
        for period in [14, 21]:
            delta = df['close'].diff()
            gain = delta.where(delta > 0, 0).rolling(window=period, min_periods=1).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period, min_periods=1).mean()
            rs = gain / (loss + eps)
            features[f'rsi_{period}'] = (100 - (100 / (1 + rs))).clip(0, 100)

        # 4. 볼린저 밴드 (안전한 계산)
        for period in [20]:
            ma = df['close'].rolling(window=period, min_periods=1).mean()
            std = df['close'].rolling(window=period, min_periods=1).std()
            features[f'bb_{period}_position'] = ((df['close'] - ma) / (2 * std + eps)).clip(-3, 3)

        # 5. 볼륨 지표 (안전한 계산)
        vol_ma = df['volume'].rolling(window=20, min_periods=1).mean()
        features['volume_ratio'] = (df['volume'] / (vol_ma + eps)).clip(0, 5)

        # 6. 변동성 (ATR)
        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift())
        low_close = abs(df['low'] - df['close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        features['atr_14'] = (tr.rolling(window=14, min_periods=1).mean() / (df['close'] + eps)).clip(0, 0.1)

        # 7. 추세 강도
        for period in [10, 20]:
            highest = df['high'].rolling(window=period, min_periods=1).max()
            lowest = df['low'].rolling(window=period, min_periods=1).min()
            features[f'trend_strength_{period}'] = ((df['close'] - lowest) / (highest - lowest + eps)).clip(0, 1)

        # 8. MACD (안전한 계산)
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9, adjust=False).mean()
        features['macd'] = (macd / (df['close'] + eps)).clip(-0.1, 0.1)
        features['macd_signal'] = (signal / (df['close'] + eps)).clip(-0.1, 0.1)

        # NaN과 infinity 제거
        features = features.replace([np.inf, -np.inf], np.nan)
        features = features.fillna(0)

        # 극단값 추가 제한
        for col in features.columns:
            q1 = features[col].quantile(0.01)
            q99 = features[col].quantile(0.99)
            features[col] = features[col].clip(q1, q99)

        return features

    def create_balanced_labels(self, df, timeframe):
        """균형잡힌 라벨 생성"""
        thresholds = {
            '15m': 0.0010,  # 0.1%
            '30m': 0.0015,  # 0.15%
            '1h': 0.0020,   # 0.2%
            '4h': 0.0030    # 0.3%
        }

        threshold = thresholds.get(timeframe, 0.002)

        # 미래 수익률 (다음 3개 캔들의 가중평균)
        future_returns = []
        weights = [0.5, 0.3, 0.2]

        for i, w in enumerate(weights, 1):
            ret = df['close'].shift(-i).pct_change()
            future_returns.append(ret * w)

        weighted_return = sum(future_returns)

        # 라벨 생성
        labels = pd.Series(index=df.index, dtype=int)
        labels[weighted_return > threshold] = 1  # UP
        labels[weighted_return < -threshold] = 0  # DOWN

        # 중립 구간 제거 (명확한 신호만)
        labels[(weighted_return >= -threshold) & (weighted_return <= threshold)] = np.nan
        labels = labels.dropna()

        return labels

    def train_optimized_model(self, timeframe):
        """최적화된 모델 훈련"""
        print(f"\n{'='*60}")
        print(f"🚀 {timeframe} 개선된 모델 훈련")
        print(f"{'='*60}")

        # 확장 데이터 수집 (90일)
        df = self.get_extended_data(timeframe, days=90)

        if len(df) < 100:
            print("  ⚠️ 데이터 부족")
            return None

        # 안전한 특징 생성
        print("  📐 안전한 특징 생성 중...")
        features = self.create_robust_features(df)

        # 균형잡힌 라벨
        labels = self.create_balanced_labels(df, timeframe)

        # 유효 데이터
        valid_idx = features.index.intersection(labels.index)
        X = features.loc[valid_idx]
        y = labels.loc[valid_idx]

        # 최종 정리
        valid_mask = ~(X.isna().any(axis=1) | y.isna())
        X = X[valid_mask]
        y = y[valid_mask]

        print(f"  📊 데이터: {len(X)}개 샘플")
        print(f"  📈 UP: {(y==1).sum()}개 ({(y==1).sum()/len(y)*100:.1f}%)")
        print(f"  📉 DOWN: {(y==0).sum()}개 ({(y==0).sum()/len(y)*100:.1f}%)")

        if len(X) < 100:
            print("  ⚠️ 유효 데이터 부족")
            return None

        # 데이터 분할
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        # RobustScaler 사용 (이상치에 강함)
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # GridSearch에서 찾은 최적 파라미터 사용
        models = {
            'xgboost': XGBClassifier(
                n_estimators=500,
                max_depth=7,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                use_label_encoder=False,
                eval_metric='logloss'
            ),
            'lightgbm': LGBMClassifier(
                n_estimators=500,
                max_depth=7,
                learning_rate=0.05,
                num_leaves=50,
                min_child_samples=10,
                random_state=42,
                verbosity=-1,
                force_col_wise=True
            ),
            'rf': RandomForestClassifier(
                n_estimators=500,
                max_depth=15,
                min_samples_split=10,
                min_samples_leaf=5,
                max_features='sqrt',
                random_state=42,
                n_jobs=-1
            ),
            'gb': GradientBoostingClassifier(
                n_estimators=300,
                max_depth=7,
                learning_rate=0.05,
                min_samples_split=20,
                min_samples_leaf=10,
                subsample=0.8,
                random_state=42
            )
        }

        # 모델 훈련
        print("\n  📊 모델 훈련:")
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

        # 앙상블 (가중 평균)
        print("\n  🎯 앙상블 예측:")

        ensemble_proba = np.zeros((len(X_test_scaled), 2))
        weights = {'xgboost': 1.5, 'lightgbm': 1.5, 'rf': 1.2, 'gb': 1.0}

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
        if len(cm) > 1:
            print(f"  실제UP      {cm[1,0]:4d}    {cm[1,1]:4d}")

        # 모델 저장
        if ensemble_acc > 0.55:
            model_info = {
                'models': {name: r['model'] for name, r in results.items()},
                'scaler': scaler,
                'features': list(features.columns),
                'ensemble_accuracy': ensemble_acc,
                'ensemble_precision': ensemble_precision,
                'ensemble_recall': ensemble_recall,
                'ensemble_f1': ensemble_f1,
                'timeframe': timeframe,
                'data_size': len(X),
                'trained_at': datetime.now().isoformat()
            }

            filename = f"improved_{timeframe}_model.pkl"
            joblib.dump(model_info, f"models/{filename}")
            print(f"\n  ✅ 모델 저장: models/{filename}")

            return model_info
        else:
            print(f"\n  ⚠️ 정확도 부족 ({ensemble_acc:.1%})")
            return None

def main():
    trainer = ImprovedAccuracyTrainer()

    print("="*60)
    print("🔧 개선된 고정확도 모델 훈련")
    print("📊 90일 데이터 사용")
    print("="*60)

    results = {}

    for timeframe in ['15m', '30m', '1h', '4h']:
        try:
            model_info = trainer.train_optimized_model(timeframe)
            if model_info:
                results[timeframe] = model_info
        except Exception as e:
            print(f"\n❌ {timeframe} 훈련 실패: {e}")

    # 결과 요약
    if results:
        print("\n" + "="*60)
        print("📋 훈련 결과 요약")
        print("="*60)

        for tf, info in results.items():
            print(f"\n{tf}:")
            print(f"  앙상블 정확도: {info['ensemble_accuracy']*100:.1f}%")
            print(f"  F1 점수: {info['ensemble_f1']:.3f}")
            print(f"  데이터 크기: {info['data_size']}개")

if __name__ == "__main__":
    main()