#!/usr/bin/env python3
"""
GPU-Based Transformer Model for BTC Direction Prediction
시간대별 상승/하락 특화 모델 - 충분한 데이터와 완전한 구현
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import ccxt
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import TimeSeriesSplit
from datetime import datetime
import joblib
import warnings
warnings.filterwarnings('ignore')

# GPU 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🖥️ Using device: {device}")
if torch.cuda.is_available():
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

class BTCDataset(Dataset):
    """BTC 시계열 데이터셋"""
    def __init__(self, X, y, seq_length=100):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        self.seq_length = seq_length

    def __len__(self):
        return len(self.X) - self.seq_length

    def __getitem__(self, idx):
        return (self.X[idx:idx+self.seq_length],
                self.y[idx+self.seq_length])

class TransformerModel(nn.Module):
    """Advanced Transformer for Time Series Prediction"""
    def __init__(self, input_dim, d_model=256, nhead=8, num_layers=6, dropout=0.2):
        super(TransformerModel, self).__init__()

        # Input projection
        self.input_projection = nn.Linear(input_dim, d_model)

        # Positional encoding
        self.positional_encoding = nn.Parameter(torch.randn(1, 500, d_model))

        # Transformer encoder
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)

        # Output layers
        self.fc1 = nn.Linear(d_model, d_model // 2)
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(d_model // 2, d_model // 4)
        self.dropout2 = nn.Dropout(dropout)
        self.fc3 = nn.Linear(d_model // 4, 1)

        # Activation
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Input shape: (batch, seq_len, features)
        batch_size, seq_len, _ = x.shape

        # Project input
        x = self.input_projection(x)

        # Add positional encoding
        x = x + self.positional_encoding[:, :seq_len, :]

        # Transformer encoding
        x = self.transformer_encoder(x)

        # Take the last output
        x = x[:, -1, :]

        # Output layers
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout2(x)

        x = self.fc3(x)
        x = self.sigmoid(x)

        return x.squeeze()

class GPUTransformerTrainer:
    def __init__(self):
        self.exchange = ccxt.binance()
        self.device = device

    def get_massive_data(self, timeframe, days=60):
        """대량 데이터 수집 (60일)"""
        print(f"📊 {timeframe} 대량 데이터 수집 ({days}일)...")

        tf_ms = {
            '15m': 15 * 60 * 1000,
            '30m': 30 * 60 * 1000,
            '1h': 60 * 60 * 1000,
            '4h': 4 * 60 * 60 * 1000,
        }

        ms_per_candle = tf_ms.get(timeframe, 60 * 60 * 1000)
        total_candles = int(days * 24 * 60 * 60 * 1000 / ms_per_candle)

        all_data = []
        chunk_size = 1000
        end_time = self.exchange.milliseconds()

        print(f"  목표: {total_candles}개 캔들")

        while len(all_data) < total_candles:
            try:
                since = end_time - len(all_data) * ms_per_candle - chunk_size * ms_per_candle
                ohlcv = self.exchange.fetch_ohlcv('BTC/USDT', timeframe, since=since, limit=chunk_size)

                if not ohlcv:
                    break

                all_data = ohlcv + all_data
                print(f"  수집 진행: {len(all_data)}/{total_candles} ({len(all_data)/total_candles*100:.1f}%)")

                if len(all_data) >= total_candles:
                    all_data = all_data[-total_candles:]
                    break

            except Exception as e:
                print(f"  ⚠️ 수집 오류: {e}")
                break

        df = pd.DataFrame(all_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)

        print(f"  ✅ {len(df)}개 캔들 수집 완료")
        return df

    def create_advanced_features(self, df, direction='up'):
        """고급 특징 생성 (방향 특화)"""
        features = pd.DataFrame(index=df.index)

        # 1. 다양한 기간의 수익률
        for period in [1, 2, 3, 5, 8, 13, 21, 34, 55, 89]:
            features[f'return_{period}'] = df['close'].pct_change(period)
            features[f'return_{period}_abs'] = features[f'return_{period}'].abs()

        # 2. 기술적 지표 (다양한 기간)
        for period in [7, 14, 21, 28]:
            delta = df['close'].diff()
            gain = delta.where(delta > 0, 0).rolling(period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
            rsi = 100 - (100 / (1 + gain / (loss + 1e-10)))
            features[f'rsi_{period}'] = rsi
            features[f'rsi_{period}_ma'] = rsi.rolling(10).mean()

        # 3. 이동평균과 트렌드
        for period in [5, 10, 20, 50, 100, 200]:
            sma = df['close'].rolling(window=period).mean()
            ema = df['close'].ewm(span=period, adjust=False).mean()
            features[f'sma_{period}_ratio'] = df['close'] / (sma + 1e-10)
            features[f'ema_{period}_ratio'] = df['close'] / (ema + 1e-10)
            features[f'sma_{period}_slope'] = sma.pct_change(5)

        # 4. 볼린저 밴드
        for period in [20, 30]:
            ma = df['close'].rolling(window=period).mean()
            std = df['close'].rolling(window=period).std()
            features[f'bb_position_{period}'] = (df['close'] - ma) / (2*std + 1e-10)
            features[f'bb_width_{period}'] = (4*std) / (ma + 1e-10)

        # 5. 볼륨 분석
        features['volume_ratio'] = df['volume'] / df['volume'].rolling(window=20).mean()
        features['volume_trend'] = df['volume'].rolling(window=20).apply(
            lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0
        )

        # 6. 가격 위치
        for period in [20, 50, 100]:
            highest = df['high'].rolling(window=period).max()
            lowest = df['low'].rolling(window=period).min()
            features[f'price_position_{period}'] = (df['close'] - lowest) / (highest - lowest + 1e-10)

        # 7. 변동성
        for period in [10, 20, 50]:
            features[f'volatility_{period}'] = df['close'].pct_change().rolling(window=period).std()
            features[f'atr_{period}'] = self.calculate_atr(df, period)

        # 8. 마켓 마이크로구조
        features['spread'] = (df['high'] - df['low']) / df['close']
        features['close_position'] = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-10)

        # 9. 방향별 특화 특징
        if direction == 'up':
            # 상승 특화
            features['higher_highs'] = (df['high'] > df['high'].shift(1)).rolling(10).sum()
            features['bullish_candles'] = ((df['close'] > df['open']) * 1.0).rolling(10).mean()
            features['up_momentum'] = df['close'].pct_change().where(lambda x: x > 0, 0).rolling(20).sum()
            features['support_bounces'] = ((df['low'] <= df['low'].rolling(20).min()) &
                                          (df['close'] > df['open'])).rolling(20).sum()
        else:
            # 하락 특화
            features['lower_lows'] = (df['low'] < df['low'].shift(1)).rolling(10).sum()
            features['bearish_candles'] = ((df['close'] < df['open']) * 1.0).rolling(10).mean()
            features['down_momentum'] = df['close'].pct_change().where(lambda x: x < 0, 0).rolling(20).sum().abs()
            features['resistance_rejections'] = ((df['high'] >= df['high'].rolling(20).max()) &
                                                (df['close'] < df['open'])).rolling(20).sum()

        # 10. 시간 특징
        features['hour'] = df.index.hour
        features['day_of_week'] = df.index.dayofweek
        features['is_asia'] = ((df.index.hour >= 0) & (df.index.hour < 8)).astype(int)
        features['is_europe'] = ((df.index.hour >= 8) & (df.index.hour < 16)).astype(int)
        features['is_us'] = ((df.index.hour >= 16) & (df.index.hour < 24)).astype(int)

        # Clean data
        features = features.replace([np.inf, -np.inf], np.nan)
        features = features.fillna(0)

        return features

    def calculate_atr(self, df, period):
        """ATR 계산"""
        high_low = df['high'] - df['low']
        high_close = (df['high'] - df['close'].shift()).abs()
        low_close = (df['low'] - df['close'].shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean() / df['close']
        return atr

    def create_labels(self, df, direction='up', horizon=8):
        """방향별 라벨 생성"""
        future_return = df['close'].shift(-horizon) / df['close'] - 1

        if direction == 'up':
            # 상승 모델: 0.3% 이상 상승
            labels = (future_return > 0.003).astype(float)
        else:
            # 하락 모델: 0.3% 이상 하락
            labels = (future_return < -0.003).astype(float)

        return labels

    def train_specialist_model(self, timeframe, direction='up'):
        """시간대/방향별 특화 모델 훈련"""
        print(f"\n{'='*60}")
        print(f"🚀 {timeframe} {direction.upper()} Transformer 모델 훈련 (GPU)")
        print(f"{'='*60}")

        # 1. 대량 데이터 수집
        df = self.get_massive_data(timeframe, days=60)

        # 2. 특징 생성
        print(f"  📐 고급 특징 생성 중...")
        features = self.create_advanced_features(df, direction)
        labels = self.create_labels(df, direction)

        # 3. 데이터 정리
        valid_idx = ~(features.isna().any(axis=1) | labels.isna())
        X = features[valid_idx].values
        y = labels[valid_idx].values

        print(f"  📊 데이터: {len(X)}개 샘플, {X.shape[1]}개 특징")
        print(f"  📈 타겟 비율: {y.mean():.1%}")

        # 4. Train/Test Split
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        # 5. 스케일링
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # 6. 데이터셋 생성
        seq_length = 50
        train_dataset = BTCDataset(X_train_scaled, y_train, seq_length)
        test_dataset = BTCDataset(X_test_scaled, y_test, seq_length)

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        # 7. 모델 초기화
        print(f"  🤖 Transformer 모델 초기화...")
        model = TransformerModel(
            input_dim=X.shape[1],
            d_model=256,
            nhead=8,
            num_layers=6,
            dropout=0.2
        ).to(self.device)

        # 8. 훈련 설정
        criterion = nn.BCELoss()
        optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.01)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

        # 9. 훈련
        print(f"  🔥 GPU 훈련 시작...")
        best_accuracy = 0
        patience = 10
        patience_counter = 0

        for epoch in range(100):
            # Training
            model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0

            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                optimizer.step()

                train_loss += loss.item()
                predictions = (outputs > 0.5).float()
                train_correct += (predictions == batch_y).sum().item()
                train_total += batch_y.size(0)

            # Validation
            model.eval()
            test_correct = 0
            test_total = 0

            with torch.no_grad():
                for batch_x, batch_y in test_loader:
                    batch_x = batch_x.to(self.device)
                    batch_y = batch_y.to(self.device)

                    outputs = model(batch_x)
                    predictions = (outputs > 0.5).float()
                    test_correct += (predictions == batch_y).sum().item()
                    test_total += batch_y.size(0)

            train_accuracy = train_correct / train_total * 100
            test_accuracy = test_correct / test_total * 100

            if (epoch + 1) % 10 == 0:
                print(f"    Epoch {epoch+1}: Train Acc={train_accuracy:.1f}%, Test Acc={test_accuracy:.1f}%")

            # Early stopping
            if test_accuracy > best_accuracy:
                best_accuracy = test_accuracy
                best_model_state = model.state_dict()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"    Early stopping at epoch {epoch+1}")
                    break

            scheduler.step()

        # 10. 최종 평가
        model.load_state_dict(best_model_state)
        model.eval()

        all_predictions = []
        all_targets = []
        all_probabilities = []

        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x = batch_x.to(self.device)
                outputs = model(batch_x)
                predictions = (outputs > 0.5).float()

                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(batch_y.numpy())
                all_probabilities.extend(outputs.cpu().numpy())

        final_accuracy = sum(p == t for p, t in zip(all_predictions, all_targets)) / len(all_targets) * 100

        print(f"\n  🏆 최종 테스트 정확도: {final_accuracy:.1f}%")
        print(f"  📊 예측 분포: {np.mean(all_predictions):.1%} positive")
        print(f"  📊 평균 신뢰도: {np.mean(np.maximum(all_probabilities, 1-np.array(all_probabilities))):.1%}")

        # 11. 모델 저장
        model_info = {
            'model_state': best_model_state,
            'model_config': {
                'input_dim': X.shape[1],
                'd_model': 256,
                'nhead': 8,
                'num_layers': 6,
                'dropout': 0.2
            },
            'scaler': scaler,
            'features': list(features.columns),
            'timeframe': timeframe,
            'direction': direction,
            'accuracy': final_accuracy,
            'seq_length': seq_length,
            'trained_at': datetime.now().isoformat()
        }

        filename = f"gpu_transformer_{timeframe}_{direction}_model.pkl"
        joblib.dump(model_info, f"models/{filename}")
        print(f"  ✅ 모델 저장: models/{filename}")

        return {
            'accuracy': final_accuracy,
            'timeframe': timeframe,
            'direction': direction
        }

def main():
    print("="*60)
    print("🚀 GPU Transformer 모델 훈련")
    print("📅 " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("🖥️ 시간대별 상승/하락 특화 모델")
    print("="*60)

    trainer = GPUTransformerTrainer()
    results = []

    # 각 시간대별로 상승/하락 모델 훈련
    for timeframe in ['15m', '30m', '1h', '4h']:
        for direction in ['up', 'down']:
            try:
                result = trainer.train_specialist_model(timeframe, direction)
                results.append(result)
            except Exception as e:
                print(f"❌ {timeframe} {direction} 훈련 실패: {e}")

    # 결과 요약
    print("\n" + "="*60)
    print("📊 GPU Transformer 훈련 결과")
    print("="*60)

    for result in results:
        emoji = "✅" if result['accuracy'] >= 60 else "⚠️" if result['accuracy'] >= 55 else "❌"
        print(f"{emoji} {result['timeframe']} {result['direction'].upper()}: {result['accuracy']:.1f}%")

    # 평가
    avg_accuracy = np.mean([r['accuracy'] for r in results])
    print(f"\n평균 정확도: {avg_accuracy:.1f}%")

    if avg_accuracy >= 60:
        print("🎉 목표 달성! 60% 이상 정확도")
    else:
        print("📈 추가 개선 필요")

if __name__ == "__main__":
    main()