#!/usr/bin/env python3
"""
Advanced ML Models for BTC Direction Prediction
2025년 12월 최신 기법 적용
- Transformer 기반 시계열 예측
- Ensemble with Stacking
- Advanced Feature Engineering
- Optuna 하이퍼파라미터 최적화
"""

import ccxt
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier

# Advanced Models
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# Deep Learning
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Feature Selection & Optimization
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

import joblib
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class TransformerModel(nn.Module):
    """Transformer 기반 시계열 예측 모델"""
    def __init__(self, input_dim, d_model=64, nhead=4, num_layers=2, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.input_projection = nn.Linear(input_dim, d_model)
        self.positional_encoding = nn.Parameter(torch.randn(1, 100, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.fc1 = nn.Linear(d_model, 32)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(32, 2)  # Binary classification
        self.relu = nn.ReLU()

    def forward(self, x):
        # x shape: (batch, seq_len, features)
        x = self.input_projection(x)
        x = x + self.positional_encoding[:, :x.size(1), :]
        x = self.transformer(x)
        x = x.mean(dim=1)  # Global average pooling
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class AdvancedMLTrainer:
    def __init__(self):
        self.exchange = ccxt.binance()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🔧 Using device: {self.device}")

    def get_data(self, timeframe, days=90):
        """확장된 데이터 수집"""
        print(f"📊 {timeframe} {days}일 데이터 수집...")

        all_data = []
        chunk_size = 1000

        tf_ms = {
            '15m': 15 * 60 * 1000,
            '30m': 30 * 60 * 1000,
            '1h': 60 * 60 * 1000,
            '4h': 240 * 60 * 1000
        }

        ms_per_candle = tf_ms.get(timeframe, 60 * 60 * 1000)
        total_candles = int(days * 24 * 60 * 60 * 1000 / ms_per_candle)

        end_time = self.exchange.milliseconds()
        current_time = end_time

        while len(all_data) < total_candles:
            try:
                ohlcv = self.exchange.fetch_ohlcv(
                    'BTC/USDT',
                    timeframe,
                    limit=chunk_size,
                    since=current_time - (chunk_size * ms_per_candle)
                )

                if not ohlcv:
                    break

                all_data = ohlcv + all_data
                current_time = ohlcv[0][0] if ohlcv else current_time

                if len(all_data) >= total_candles:
                    all_data = all_data[-total_candles:]
                    break

            except Exception as e:
                print(f"  ⚠️ 수집 중단: {e}")
                break

        df = pd.DataFrame(all_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)

        print(f"  ✅ {len(df)}개 캔들 수집 완료")
        return df

    def create_advanced_features(self, df, direction='up'):
        """2025년 최신 특징 엔지니어링"""
        features = pd.DataFrame(index=df.index)

        # 1. Price Action Features
        features['log_return'] = np.log(df['close'] / df['close'].shift(1))
        for lag in [1, 2, 3, 5, 8, 13, 21]:
            features[f'return_lag_{lag}'] = df['close'].pct_change(lag)

        # 2. Volatility Features
        features['garman_klass'] = np.sqrt(
            0.5 * np.log(df['high'] / df['low'])**2 -
            (2 * np.log(2) - 1) * np.log(df['close'] / df['open'])**2
        )

        features['parkinson'] = np.sqrt(
            np.log(df['high'] / df['low'])**2 / (4 * np.log(2))
        )

        # Rolling volatility
        for window in [5, 10, 20]:
            features[f'volatility_{window}'] = features['log_return'].rolling(window).std()

        # 3. Microstructure Features
        features['bid_ask_proxy'] = 2 * np.sqrt(np.abs(
            np.log(df['high'] / df['close']) * np.log(df['high'] / df['open'])
        ))

        features['kyle_lambda'] = (df['close'] - df['open']) / (df['volume'] + 1e-10)
        features['amihud_illiquidity'] = np.abs(features['log_return']) / (df['volume'] + 1e-10)

        # 4. Order Flow Imbalance (proxy)
        features['voi'] = df['volume'] * np.sign(df['close'] - df['open'])
        features['voi_ratio'] = features['voi'].rolling(10).sum() / (df['volume'].rolling(10).sum() + 1e-10)

        # 5. Technical Indicators (Advanced)
        # RSI with multiple periods
        for period in [7, 14, 21]:
            delta = df['close'].diff()
            gain = delta.where(delta > 0, 0).rolling(period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
            features[f'rsi_{period}'] = 100 - (100 / (1 + gain / (loss + 1e-10)))

        # MACD variations
        for fast, slow in [(12, 26), (5, 20), (8, 21)]:
            exp1 = df['close'].ewm(span=fast, adjust=False).mean()
            exp2 = df['close'].ewm(span=slow, adjust=False).mean()
            features[f'macd_{fast}_{slow}'] = (exp1 - exp2) / df['close']

        # Bollinger Bands features
        for period in [20, 50]:
            ma = df['close'].rolling(period).mean()
            std = df['close'].rolling(period).std()
            features[f'bb_position_{period}'] = (df['close'] - ma) / (2 * std + 1e-10)
            features[f'bb_width_{period}'] = std / ma

        # 6. Volume Profile
        features['volume_ma_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
        features['volume_std'] = df['volume'].rolling(20).std() / df['volume'].rolling(20).mean()

        # VWAP
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        features['vwap'] = (typical_price * df['volume']).rolling(20).sum() / df['volume'].rolling(20).sum()
        features['vwap_distance'] = (df['close'] - features['vwap']) / features['vwap']

        # 7. Market Regime Features
        features['trend_strength'] = df['close'].rolling(20).apply(
            lambda x: np.polyfit(range(len(x)), x, 1)[0] / (np.std(x) + 1e-10)
        )

        # Hurst Exponent (simplified)
        def hurst(ts):
            lags = range(2, min(20, len(ts)//2))
            tau = [np.std(np.subtract(ts[lag:], ts[:-lag])) for lag in lags]
            poly = np.polyfit(np.log(lags), np.log(tau), 1)
            return poly[0] * 2.0

        features['hurst'] = df['close'].rolling(50, min_periods=20).apply(hurst, raw=False)

        # 8. Directional Features
        if direction == 'up':
            # Bullish patterns
            features['higher_highs'] = (df['high'] > df['high'].shift(1)).rolling(5).sum()
            features['higher_lows'] = (df['low'] > df['low'].shift(1)).rolling(5).sum()
            features['bullish_engulfing'] = (
                (df['close'] > df['open']) &
                (df['close'].shift(1) < df['open'].shift(1)) &
                (df['close'] > df['open'].shift(1)) &
                (df['open'] < df['close'].shift(1))
            ).astype(int)

        else:  # down
            # Bearish patterns
            features['lower_highs'] = (df['high'] < df['high'].shift(1)).rolling(5).sum()
            features['lower_lows'] = (df['low'] < df['low'].shift(1)).rolling(5).sum()
            features['bearish_engulfing'] = (
                (df['close'] < df['open']) &
                (df['close'].shift(1) > df['open'].shift(1)) &
                (df['close'] < df['open'].shift(1)) &
                (df['open'] > df['close'].shift(1))
            ).astype(int)

        # 9. Cyclical Features
        features['hour'] = df.index.hour
        features['day_of_week'] = df.index.dayofweek
        features['hour_sin'] = np.sin(2 * np.pi * features['hour'] / 24)
        features['hour_cos'] = np.cos(2 * np.pi * features['hour'] / 24)

        # 10. Interaction Features
        features['rsi_volume'] = features['rsi_14'] * features['volume_ma_ratio']
        features['volatility_volume'] = features['volatility_10'] * features['volume_ma_ratio']
        features['trend_volatility'] = features['trend_strength'] * features['volatility_20']

        # Clean up
        features = features.replace([np.inf, -np.inf], np.nan)
        features = features.fillna(method='ffill').fillna(0)

        # Clip extreme values
        for col in features.columns:
            q1 = features[col].quantile(0.01)
            q99 = features[col].quantile(0.99)
            features[col] = features[col].clip(q1, q99)

        return features

    def create_labels(self, df, direction='up', timeframe='15m'):
        """방향별 라벨 생성"""
        # 타임프레임별 예측 horizon
        horizons = {
            '15m': 4,   # 1시간 후
            '30m': 4,   # 2시간 후
            '1h': 2,    # 2시간 후
            '4h': 1     # 4시간 후
        }

        horizon = horizons.get(timeframe, 2)

        # 미래 수익률
        future_return = df['close'].shift(-horizon) / df['close'] - 1

        # 타임프레임별 임계값
        thresholds = {
            '15m': 0.0015,  # 0.15%
            '30m': 0.0020,  # 0.20%
            '1h': 0.0025,   # 0.25%
            '4h': 0.0040    # 0.40%
        }

        threshold = thresholds.get(timeframe, 0.002)

        if direction == 'up':
            # 상승 예측: threshold 이상 상승
            labels = (future_return > threshold).astype(int)
        else:
            # 하락 예측: threshold 이상 하락
            labels = (future_return < -threshold).astype(int)

        return labels

    def optimize_hyperparameters(self, X_train, y_train, model_type='xgboost'):
        """Optuna를 사용한 하이퍼파라미터 최적화"""

        def objective(trial):
            if model_type == 'xgboost':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'gamma': trial.suggest_float('gamma', 0, 5),
                    'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
                    'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
                    'random_state': 42
                }
                model = XGBClassifier(**params, use_label_encoder=False, eval_metric='logloss')

            elif model_type == 'lightgbm':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                    'num_leaves': trial.suggest_int('num_leaves', 10, 100),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
                    'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
                    'random_state': 42,
                    'verbosity': -1
                }
                model = LGBMClassifier(**params)

            elif model_type == 'catboost':
                params = {
                    'iterations': trial.suggest_int('iterations', 100, 1000),
                    'depth': trial.suggest_int('depth', 4, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
                    'random_seed': 42,
                    'verbose': False
                }
                model = CatBoostClassifier(**params)

            # Cross-validation
            tscv = TimeSeriesSplit(n_splits=3)
            scores = []

            for train_idx, val_idx in tscv.split(X_train):
                X_tr, X_val = X_train[train_idx], X_train[val_idx]
                y_tr, y_val = y_train[train_idx], y_train[val_idx]

                model.fit(X_tr, y_tr)
                y_pred = model.predict(X_val)
                scores.append(accuracy_score(y_val, y_pred))

            return np.mean(scores)

        # Optimize
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=20, show_progress_bar=False)

        return study.best_params

    def train_transformer_model(self, X_train, y_train, X_test, y_test, input_dim):
        """Transformer 모델 훈련"""
        # Prepare sequences
        seq_len = min(20, len(X_train) // 10)  # Sequence length

        X_train_seq = []
        y_train_seq = []

        for i in range(len(X_train) - seq_len):
            X_train_seq.append(X_train[i:i+seq_len])
            y_train_seq.append(y_train[i+seq_len-1])

        X_train_seq = np.array(X_train_seq)
        y_train_seq = np.array(y_train_seq)

        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train_seq).to(self.device)
        y_train_tensor = torch.LongTensor(y_train_seq).to(self.device)

        # Create DataLoader
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)

        # Initialize model
        model = TransformerModel(input_dim).to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # Train
        model.train()
        for epoch in range(50):
            total_loss = 0
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

        # Evaluate
        model.eval()
        with torch.no_grad():
            # Prepare test sequences
            X_test_seq = []
            y_test_seq = []

            for i in range(len(X_test) - seq_len):
                X_test_seq.append(X_test[i:i+seq_len])
                y_test_seq.append(y_test[i+seq_len-1])

            if len(X_test_seq) > 0:
                X_test_tensor = torch.FloatTensor(np.array(X_test_seq)).to(self.device)
                outputs = model(X_test_tensor)
                _, predicted = torch.max(outputs, 1)
                accuracy = accuracy_score(y_test_seq, predicted.cpu().numpy())
            else:
                accuracy = 0

        return model, accuracy

    def train_specialist_model(self, timeframe, direction='up'):
        """전문화 모델 훈련"""
        print(f"\n{'='*60}")
        print(f"🚀 {timeframe} {direction.upper()} 전문 모델 훈련")
        print(f"{'='*60}")

        # 데이터 수집
        df = self.get_data(timeframe, days=90)

        # 특징 생성
        print(f"  📐 Advanced 특징 생성 중...")
        features = self.create_advanced_features(df, direction)

        # 라벨 생성
        labels = self.create_labels(df, direction, timeframe)

        # 유효 데이터
        valid_idx = ~(features.isna().any(axis=1) | labels.isna())
        X = features[valid_idx].values
        y = labels[valid_idx].values

        print(f"  📊 데이터: {len(X)}개 샘플, {X.shape[1]}개 특징")
        print(f"  📈 타겟 비율: {y.mean():.1%}")

        # Feature selection
        print(f"  🔍 특징 선택 중...")
        selector = SelectKBest(f_classif, k=min(50, X.shape[1]))
        X_selected = selector.fit_transform(X, y)
        selected_features = features.columns[selector.get_support()].tolist()

        # Train-test split
        split_idx = int(len(X_selected) * 0.8)
        X_train, X_test = X_selected[:split_idx], X_selected[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        # Scaling
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        results = {}

        # 1. XGBoost with Optuna
        print(f"  🔧 XGBoost 최적화 중...")
        xgb_params = self.optimize_hyperparameters(X_train_scaled, y_train, 'xgboost')
        xgb_model = XGBClassifier(**xgb_params, use_label_encoder=False, eval_metric='logloss')
        xgb_model.fit(X_train_scaled, y_train)
        xgb_pred = xgb_model.predict(X_test_scaled)
        xgb_acc = accuracy_score(y_test, xgb_pred)
        results['xgboost'] = {'model': xgb_model, 'accuracy': xgb_acc}
        print(f"    XGBoost 정확도: {xgb_acc:.1%}")

        # 2. LightGBM with Optuna
        print(f"  🔧 LightGBM 최적화 중...")
        lgb_params = self.optimize_hyperparameters(X_train_scaled, y_train, 'lightgbm')
        lgb_model = LGBMClassifier(**lgb_params)
        lgb_model.fit(X_train_scaled, y_train)
        lgb_pred = lgb_model.predict(X_test_scaled)
        lgb_acc = accuracy_score(y_test, lgb_pred)
        results['lightgbm'] = {'model': lgb_model, 'accuracy': lgb_acc}
        print(f"    LightGBM 정확도: {lgb_acc:.1%}")

        # 3. CatBoost with Optuna
        print(f"  🔧 CatBoost 최적화 중...")
        cat_params = self.optimize_hyperparameters(X_train_scaled, y_train, 'catboost')
        cat_model = CatBoostClassifier(**cat_params)
        cat_model.fit(X_train_scaled, y_train, verbose=False)
        cat_pred = cat_model.predict(X_test_scaled)
        cat_acc = accuracy_score(y_test, cat_pred)
        results['catboost'] = {'model': cat_model, 'accuracy': cat_acc}
        print(f"    CatBoost 정확도: {cat_acc:.1%}")

        # 4. Neural Network
        print(f"  🧠 Neural Network 훈련 중...")
        nn_model = MLPClassifier(
            hidden_layer_sizes=(128, 64, 32),
            activation='relu',
            solver='adam',
            alpha=0.001,
            batch_size=32,
            learning_rate='adaptive',
            max_iter=500,
            random_state=42,
            early_stopping=True
        )
        nn_model.fit(X_train_scaled, y_train)
        nn_pred = nn_model.predict(X_test_scaled)
        nn_acc = accuracy_score(y_test, nn_pred)
        results['neural_network'] = {'model': nn_model, 'accuracy': nn_acc}
        print(f"    Neural Network 정확도: {nn_acc:.1%}")

        # 5. Transformer (if enough data)
        if len(X_train_scaled) > 500:
            print(f"  🤖 Transformer 모델 훈련 중...")
            transformer_model, transformer_acc = self.train_transformer_model(
                X_train_scaled, y_train, X_test_scaled, y_test, X_selected.shape[1]
            )
            results['transformer'] = {'model': transformer_model, 'accuracy': transformer_acc}
            print(f"    Transformer 정확도: {transformer_acc:.1%}")

        # 6. Ensemble (Voting)
        print(f"  🎯 Ensemble 예측...")
        ensemble_pred = np.zeros(len(X_test_scaled))
        weights = []

        for name, result in results.items():
            if name != 'transformer':  # Transformer는 별도 처리
                model = result['model']
                pred = model.predict(X_test_scaled)
                weight = result['accuracy']
                ensemble_pred += pred * weight
                weights.append(weight)

        ensemble_pred = (ensemble_pred / sum(weights) > 0.5).astype(int)
        ensemble_acc = accuracy_score(y_test, ensemble_pred)
        print(f"    Ensemble 정확도: {ensemble_acc:.1%}")

        # Select best model
        best_model_name = max(results, key=lambda x: results[x]['accuracy'])
        best_accuracy = results[best_model_name]['accuracy']

        print(f"\n  🏆 최고 모델: {best_model_name} ({best_accuracy:.1%})")

        # Save model
        model_info = {
            'models': {k: v['model'] for k, v in results.items() if k != 'transformer'},
            'scaler': scaler,
            'selector': selector,
            'selected_features': selected_features,
            'direction': direction,
            'timeframe': timeframe,
            'best_model': best_model_name,
            'best_accuracy': best_accuracy,
            'ensemble_accuracy': ensemble_acc,
            'trained_at': datetime.now().isoformat()
        }

        filename = f"advanced_{timeframe}_{direction}_model.pkl"
        joblib.dump(model_info, f"models/{filename}")
        print(f"  ✅ 모델 저장: models/{filename}")

        return model_info

    def train_all_models(self):
        """모든 타임프레임과 방향에 대해 모델 훈련"""
        results = {}

        for timeframe in ['15m', '30m', '1h']:
            for direction in ['up', 'down']:
                try:
                    model_info = self.train_specialist_model(timeframe, direction)
                    results[f"{timeframe}_{direction}"] = model_info
                except Exception as e:
                    print(f"❌ {timeframe} {direction} 훈련 실패: {e}")

        return results

def main():
    print("="*60)
    print("🚀 Advanced ML 모델 훈련")
    print("📅 2025년 12월 최신 기법 적용")
    print("="*60)

    trainer = AdvancedMLTrainer()
    results = trainer.train_all_models()

    # 결과 요약
    print("\n" + "="*60)
    print("📊 훈련 결과 요약")
    print("="*60)

    for key, info in results.items():
        tf, direction = key.rsplit('_', 1)
        print(f"\n{tf} {direction.upper()}:")
        print(f"  최고 모델: {info['best_model']}")
        print(f"  정확도: {info['best_accuracy']*100:.1f}%")
        print(f"  앙상블 정확도: {info['ensemble_accuracy']*100:.1f}%")

if __name__ == "__main__":
    main()