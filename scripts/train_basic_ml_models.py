#!/usr/bin/env python3
"""
Basic ML Models for BTC Direction Prediction
실제 ML 모델 (model.predict() 사용) - 빠른 훈련용
"""

import ccxt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import joblib
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class BasicMLTrainer:
    def __init__(self):
        self.exchange = ccxt.binance()

    def get_data(self, timeframe, limit=3000):
        """데이터 수집"""
        print(f"📊 {timeframe} 데이터 수집 ({limit}개 캔들)...")

        ohlcv = self.exchange.fetch_ohlcv('BTC/USDT', timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)

        print(f"  ✅ {len(df)}개 캔들 수집 완료")
        return df

    def create_basic_features(self, df):
        """기본 특징 생성 (빠른 훈련용)"""
        features = pd.DataFrame(index=df.index)

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

        # Clean data
        features = features.replace([np.inf, -np.inf], np.nan)
        features = features.fillna(0)

        return features

    def create_labels(self, df, horizon=4):
        """라벨 생성 (미래 방향)"""
        # Future return
        future_return = df['close'].shift(-horizon) / df['close'] - 1

        # Binary labels (up/down)
        labels = (future_return > 0).astype(int)

        return labels

    def train_models(self, timeframe):
        """모델 훈련"""
        print(f"\n{'='*60}")
        print(f"🚀 {timeframe} ML 모델 훈련")
        print(f"{'='*60}")

        # Get data
        df = self.get_data(timeframe, limit=3000)

        # Create features and labels
        print(f"  📐 특징 생성 중...")
        features = self.create_basic_features(df)
        labels = self.create_labels(df)

        # Remove NaN
        valid_idx = ~(features.isna().any(axis=1) | labels.isna())
        X = features[valid_idx]
        y = labels[valid_idx]

        print(f"  📊 데이터: {len(X)}개 샘플, {X.shape[1]}개 특징")
        print(f"  📈 UP 비율: {y.mean():.1%}")

        # Split data (time series split)
        split_idx = int(len(X) * 0.8)
        X_train = X[:split_idx]
        X_test = X[split_idx:]
        y_train = y[:split_idx]
        y_test = y[split_idx:]

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train models
        models = {}

        # 1. XGBoost
        print(f"  🔧 XGBoost 훈련 중...")
        xgb = XGBClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss'
        )
        xgb.fit(X_train_scaled, y_train)

        # Predictions
        xgb_pred = xgb.predict(X_test_scaled)
        xgb_proba = xgb.predict_proba(X_test_scaled)[:, 1]

        # Metrics
        xgb_acc = accuracy_score(y_test, xgb_pred)
        xgb_prec = precision_score(y_test, xgb_pred, zero_division=0)
        xgb_rec = recall_score(y_test, xgb_pred, zero_division=0)
        xgb_f1 = f1_score(y_test, xgb_pred, zero_division=0)
        xgb_auc = roc_auc_score(y_test, xgb_proba)

        models['xgboost'] = {
            'model': xgb,
            'accuracy': xgb_acc,
            'precision': xgb_prec,
            'recall': xgb_rec,
            'f1': xgb_f1,
            'auc': xgb_auc
        }

        print(f"    정확도: {xgb_acc:.1%}, AUC: {xgb_auc:.3f}")

        # 2. LightGBM
        print(f"  🔧 LightGBM 훈련 중...")
        lgb = LGBMClassifier(
            n_estimators=200,
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

        # Predictions
        lgb_pred = lgb.predict(X_test_scaled)
        lgb_proba = lgb.predict_proba(X_test_scaled)[:, 1]

        # Metrics
        lgb_acc = accuracy_score(y_test, lgb_pred)
        lgb_prec = precision_score(y_test, lgb_pred, zero_division=0)
        lgb_rec = recall_score(y_test, lgb_pred, zero_division=0)
        lgb_f1 = f1_score(y_test, lgb_pred, zero_division=0)
        lgb_auc = roc_auc_score(y_test, lgb_proba)

        models['lightgbm'] = {
            'model': lgb,
            'accuracy': lgb_acc,
            'precision': lgb_prec,
            'recall': lgb_rec,
            'f1': lgb_f1,
            'auc': lgb_auc
        }

        print(f"    정확도: {lgb_acc:.1%}, AUC: {lgb_auc:.3f}")

        # 3. Random Forest
        print(f"  🔧 Random Forest 훈련 중...")
        rf = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=20,
            min_samples_leaf=10,
            random_state=42,
            n_jobs=-1
        )
        rf.fit(X_train_scaled, y_train)

        # Predictions
        rf_pred = rf.predict(X_test_scaled)
        rf_proba = rf.predict_proba(X_test_scaled)[:, 1]

        # Metrics
        rf_acc = accuracy_score(y_test, rf_pred)
        rf_prec = precision_score(y_test, rf_pred, zero_division=0)
        rf_rec = recall_score(y_test, rf_pred, zero_division=0)
        rf_f1 = f1_score(y_test, rf_pred, zero_division=0)
        rf_auc = roc_auc_score(y_test, rf_proba)

        models['random_forest'] = {
            'model': rf,
            'accuracy': rf_acc,
            'precision': rf_prec,
            'recall': rf_rec,
            'f1': rf_f1,
            'auc': rf_auc
        }

        print(f"    정확도: {rf_acc:.1%}, AUC: {rf_auc:.3f}")

        # Ensemble
        print(f"  🎯 앙상블 예측...")
        ensemble_proba = (xgb_proba + lgb_proba + rf_proba) / 3
        ensemble_pred = (ensemble_proba > 0.5).astype(int)

        ensemble_acc = accuracy_score(y_test, ensemble_pred)
        ensemble_prec = precision_score(y_test, ensemble_pred, zero_division=0)
        ensemble_rec = recall_score(y_test, ensemble_pred, zero_division=0)
        ensemble_f1 = f1_score(y_test, ensemble_pred, zero_division=0)
        ensemble_auc = roc_auc_score(y_test, ensemble_proba)

        print(f"    앙상블 정확도: {ensemble_acc:.1%}, AUC: {ensemble_auc:.3f}")

        # Select best model
        best_model_name = max(models, key=lambda x: models[x]['accuracy'])
        best_model = models[best_model_name]

        print(f"\n  🏆 최고 모델: {best_model_name}")
        print(f"     정확도: {best_model['accuracy']:.1%}")
        print(f"     정밀도: {best_model['precision']:.1%}")
        print(f"     재현율: {best_model['recall']:.1%}")
        print(f"     F1 점수: {best_model['f1']:.3f}")
        print(f"     AUC: {best_model['auc']:.3f}")

        # Save model
        model_info = {
            'models': models,
            'scaler': scaler,
            'features': list(features.columns),
            'timeframe': timeframe,
            'best_model': best_model_name,
            'best_accuracy': best_model['accuracy'],
            'ensemble_accuracy': ensemble_acc,
            'ensemble_auc': ensemble_auc,
            'trained_at': datetime.now().isoformat()
        }

        filename = f"basic_ml_{timeframe}_model.pkl"
        joblib.dump(model_info, f"models/{filename}")
        print(f"\n  ✅ 모델 저장: models/{filename}")

        # Print test samples
        print(f"\n  📝 테스트 샘플 예측:")
        for i in range(min(5, len(y_test))):
            actual = "UP" if y_test.iloc[i] == 1 else "DOWN"
            pred = "UP" if ensemble_pred[i] == 1 else "DOWN"
            prob = ensemble_proba[i]
            status = "✅" if actual == pred else "❌"
            print(f"    {status} 실제: {actual}, 예측: {pred} (확률: {prob:.3f})")

        return model_info

def main():
    print("="*60)
    print("🚀 Basic ML 모델 훈련")
    print("📅 " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("💡 실제 model.predict() 사용")
    print("="*60)

    trainer = BasicMLTrainer()
    results = {}

    for timeframe in ['15m', '30m', '1h', '4h']:
        try:
            model_info = trainer.train_models(timeframe)
            results[timeframe] = model_info
        except Exception as e:
            print(f"❌ {timeframe} 훈련 실패: {e}")

    # Summary
    print("\n" + "="*60)
    print("📊 훈련 결과 요약")
    print("="*60)

    for tf, info in results.items():
        print(f"\n{tf}:")
        print(f"  최고 모델: {info['best_model']}")
        print(f"  정확도: {info['best_accuracy']*100:.1f}%")
        print(f"  앙상블 정확도: {info['ensemble_accuracy']*100:.1f}%")
        print(f"  AUC: {info['ensemble_auc']:.3f}")

    print("\n💡 참고:")
    print("  - 실제 ML 예측 사용 (조건 기반 아님)")
    print("  - 60% 이상 정확도가 목표")
    print("  - AUC > 0.6이면 유의미한 예측력")

if __name__ == "__main__":
    main()