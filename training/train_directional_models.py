#!/usr/bin/env python3
"""
타임프레임별 독립적인 방향성 예측 모델
- NEUTRAL 제거, UP/DOWN만 예측
- 각 타임프레임에 최적화된 전략
"""

import ccxt
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import train_test_split, TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import joblib
from datetime import datetime
from loguru import logger
import warnings
warnings.filterwarnings('ignore')

class DirectionalTradingModels:
    def __init__(self):
        self.exchange = ccxt.binance()

        # 타임프레임별 전략 설정
        self.strategies = {
            '15m': {
                'type': 'scalping',
                'lookback': 20,  # 최근 20개 캔들 참조
                'threshold': 0.0005,  # 0.05% (스캘핑용 작은 움직임)
                'data_limit': 4000,
                'features': ['momentum', 'rsi', 'volume_burst', 'micro_pattern']
            },
            '30m': {
                'type': 'swing',
                'lookback': 30,
                'threshold': 0.001,  # 0.1%
                'data_limit': 3000,
                'features': ['trend', 'macd', 'bollinger', 'volume_trend']
            },
            '4h': {
                'type': 'position',
                'lookback': 50,
                'threshold': 0.002,  # 0.2%
                'data_limit': 2000,
                'features': ['ma_cross', 'trend_strength', 'support_resistance', 'volume_profile']
            },
            '1d': {
                'type': 'trend',
                'lookback': 100,
                'threshold': 0.003,  # 0.3%
                'data_limit': 1000,
                'features': ['long_trend', 'market_structure', 'momentum_divergence', 'accumulation']
            }
        }

    def get_data(self, timeframe):
        """데이터 가져오기"""
        limit = self.strategies[timeframe]['data_limit']
        logger.info(f"  📊 데이터 수집: {timeframe} ({limit}개 캔들)")

        ohlcv = self.exchange.fetch_ohlcv('BTC/USDT', timeframe=timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)

        return df

    def create_directional_labels(self, df, timeframe):
        """방향성 라벨 생성 (UP=1, DOWN=0)"""
        strategy = self.strategies[timeframe]
        threshold = strategy['threshold']

        # 미래 수익률
        future_return = df['close'].shift(-1) / df['close'] - 1

        # 이진 분류: UP(1) or DOWN(0)
        labels = (future_return > threshold).astype(int)

        # 분포 확인
        up_count = labels.sum()
        down_count = len(labels) - up_count
        total = len(labels)

        logger.info(f"  📈 라벨 분포 ({timeframe}):")
        logger.info(f"    UP:   {up_count:4d}개 ({up_count/total*100:5.1f}%)")
        logger.info(f"    DOWN: {down_count:4d}개 ({down_count/total*100:5.1f}%)")

        return labels

    def create_15m_features(self, df):
        """15분 스캘핑 전략 특징"""
        features = pd.DataFrame(index=df.index)

        # 1. 단기 모멘텀
        features['momentum_1'] = df['close'].pct_change(1) * 100
        features['momentum_3'] = df['close'].pct_change(3) * 100
        features['momentum_5'] = df['close'].pct_change(5) * 100

        # 2. RSI (빠른 반응)
        for period in [7, 14]:
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss.replace(0, 1)
            features[f'rsi_{period}'] = 100 - (100 / (1 + rs))

        # 3. 볼륨 버스트
        features['volume_burst'] = df['volume'] / df['volume'].rolling(5).mean()
        features['volume_spike'] = (df['volume'] > df['volume'].rolling(20).mean() * 1.5).astype(int)

        # 4. 미세 패턴
        features['higher_high'] = (df['high'] > df['high'].shift(1)).astype(int)
        features['higher_low'] = (df['low'] > df['low'].shift(1)).astype(int)
        features['candle_body'] = abs(df['close'] - df['open']) / df['close'] * 100
        features['upper_shadow'] = (df['high'] - df[['close', 'open']].max(axis=1)) / df['close'] * 100

        # 5. 단기 볼린저 밴드
        sma = df['close'].rolling(10).mean()
        std = df['close'].rolling(10).std()
        features['bb_position'] = (df['close'] - sma) / (std * 2)

        return features.fillna(0)

    def create_30m_features(self, df):
        """30분 스윙 트레이딩 특징"""
        features = pd.DataFrame(index=df.index)

        # 1. 트렌드 지표
        features['trend_5'] = (df['close'] / df['close'].shift(5) - 1) * 100
        features['trend_10'] = (df['close'] / df['close'].shift(10) - 1) * 100
        features['trend_20'] = (df['close'] / df['close'].shift(20) - 1) * 100

        # 2. MACD
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        features['macd'] = exp1 - exp2
        features['macd_signal'] = features['macd'].ewm(span=9, adjust=False).mean()
        features['macd_histogram'] = features['macd'] - features['macd_signal']

        # 3. 볼린저 밴드 (표준)
        for period in [20, 30]:
            sma = df['close'].rolling(period).mean()
            std = df['close'].rolling(period).std()
            features[f'bb_width_{period}'] = (std * 2) / sma * 100
            features[f'bb_position_{period}'] = (df['close'] - sma) / (std * 2)

        # 4. 볼륨 트렌드
        features['volume_ma_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
        features['obv'] = (df['volume'] * np.sign(df['close'].diff())).cumsum()
        features['obv_ma'] = features['obv'] / features['obv'].rolling(10).mean()

        return features.fillna(0)

    def create_4h_features(self, df):
        """4시간 포지션 트레이딩 특징"""
        features = pd.DataFrame(index=df.index)

        # 1. 이동평균 크로스
        ma20 = df['close'].rolling(20).mean()
        ma50 = df['close'].rolling(50).mean()
        ma100 = df['close'].rolling(100).mean()

        features['ma20_ratio'] = df['close'] / ma20
        features['ma50_ratio'] = df['close'] / ma50
        features['ma_cross_20_50'] = (ma20 > ma50).astype(int)
        features['ma_cross_50_100'] = (ma50 > ma100).astype(int)

        # 2. 트렌드 강도
        features['trend_strength'] = abs(df['close'].pct_change(20)) * 100
        features['trend_consistency'] = (df['close'].diff().rolling(10).apply(lambda x: (x > 0).sum() / len(x)))

        # 3. 지지/저항
        features['distance_from_high'] = (df['high'].rolling(50).max() - df['close']) / df['close'] * 100
        features['distance_from_low'] = (df['close'] - df['low'].rolling(50).min()) / df['close'] * 100

        # 4. 볼륨 프로파일
        features['volume_profile'] = df['volume'].rolling(20).mean() / df['volume'].rolling(100).mean()
        features['volume_trend'] = df['volume'].rolling(20).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0])

        # 5. ATR (변동성)
        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift())
        low_close = abs(df['low'] - df['close'].shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        features['atr'] = true_range.rolling(14).mean() / df['close'] * 100

        return features.fillna(0)

    def create_1d_features(self, df):
        """1일 트렌드 팔로잉 특징"""
        features = pd.DataFrame(index=df.index)

        # 1. 장기 트렌드
        ma50 = df['close'].rolling(50).mean()
        ma100 = df['close'].rolling(100).mean()
        ma200 = df['close'].rolling(200).mean()

        features['long_trend'] = (df['close'] / ma200 - 1) * 100
        features['trend_alignment'] = ((df['close'] > ma50) & (ma50 > ma100) & (ma100 > ma200)).astype(int)

        # 2. 시장 구조
        features['higher_high_weekly'] = (df['high'].rolling(7).max() > df['high'].rolling(7).max().shift(7)).astype(int)
        features['higher_low_weekly'] = (df['low'].rolling(7).min() > df['low'].rolling(7).min().shift(7)).astype(int)

        # 3. 모멘텀 다이버전스
        rsi_14 = 100 - (100 / (1 + (df['close'].diff().where(lambda x: x > 0, 0).rolling(14).mean() /
                                   (-df['close'].diff().where(lambda x: x < 0, 0).rolling(14).mean()))))
        features['rsi_divergence'] = (df['close'].pct_change(14) * 100) - (rsi_14.pct_change(14) * 100)

        # 4. 축적/분배
        features['accumulation'] = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low']) * df['volume']
        features['accumulation_ma'] = features['accumulation'].rolling(20).mean()

        # 5. 장기 변동성
        features['volatility_30d'] = df['close'].pct_change().rolling(30).std() * 100
        features['volatility_ratio'] = features['volatility_30d'] / df['close'].pct_change().rolling(90).std() / 100

        return features.fillna(0)

    def train_model(self, timeframe):
        """타임프레임별 맞춤 모델 훈련"""
        logger.info(f"\n{'='*70}")
        logger.info(f"🎯 {timeframe} 방향성 모델 훈련")
        logger.info(f"   전략: {self.strategies[timeframe]['type'].upper()}")
        logger.info(f"{'='*70}")

        # 데이터 수집
        df = self.get_data(timeframe)

        # 특징 생성 (타임프레임별)
        if timeframe == '15m':
            features = self.create_15m_features(df)
        elif timeframe == '30m':
            features = self.create_30m_features(df)
        elif timeframe == '4h':
            features = self.create_4h_features(df)
        else:  # 1d
            features = self.create_1d_features(df)

        # 라벨 생성
        labels = self.create_directional_labels(df, timeframe)

        # 정렬
        X = features.dropna()
        y = labels[X.index][:-1]
        X = X[:-1]

        # 훈련/테스트 분할
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        # 클래스 가중치
        class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
        class_weight_dict = dict(zip(np.unique(y_train), class_weights))

        # 스케일링
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # 모델 생성 (타임프레임별 최적화)
        if timeframe in ['15m', '30m']:
            # 단기: 빠른 반응 중요
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=20,
                class_weight=class_weight_dict,
                random_state=42,
                n_jobs=-1
            )
        else:
            # 장기: 안정성 중요
            model = GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                min_samples_split=50,
                random_state=42
            )

        # 훈련
        logger.info("  🔧 모델 훈련 중...")
        model.fit(X_train_scaled, y_train)

        # 평가
        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)

        # 예측 확률
        y_pred_proba = model.predict_proba(X_test_scaled)

        # 높은 신뢰도 예측만
        high_conf_threshold = 0.65 if timeframe in ['15m', '30m'] else 0.6
        high_conf_mask = np.max(y_pred_proba, axis=1) >= high_conf_threshold

        logger.info(f"\n  📊 성능 지표:")
        logger.info(f"    전체 정확도: {accuracy*100:.1f}%")

        if high_conf_mask.any():
            high_conf_acc = accuracy_score(y_test[high_conf_mask], y_pred[high_conf_mask])
            high_conf_ratio = high_conf_mask.sum() / len(y_test) * 100

            logger.info(f"    고신뢰도 정확도: {high_conf_acc*100:.1f}% ({high_conf_ratio:.1f}% 신호)")

            # 고신뢰도 혼동 행렬
            cm = confusion_matrix(y_test[high_conf_mask], y_pred[high_conf_mask])
            if cm.shape == (2, 2):
                logger.info(f"    고신뢰도 UP 정확도: {cm[1,1]/(cm[1,0]+cm[1,1])*100:.1f}%")
                logger.info(f"    고신뢰도 DOWN 정확도: {cm[0,0]/(cm[0,0]+cm[0,1])*100:.1f}%")

        # 최근 예측 테스트
        logger.info(f"\n  🔮 최근 10개 예측:")
        recent_features = features.iloc[-11:-1]
        recent_scaled = scaler.transform(recent_features)
        recent_pred = model.predict(recent_scaled)
        recent_proba = model.predict_proba(recent_scaled)

        for i in range(len(recent_pred)):
            direction = "UP🔺" if recent_pred[i] == 1 else "DOWN🔻"
            confidence = max(recent_proba[i]) * 100
            timestamp = recent_features.index[i].strftime('%m-%d %H:%M')
            logger.info(f"    {timestamp}: {direction} ({confidence:.1f}%)")

        # 모델 저장
        model_file = f'models/directional_{timeframe}_model.pkl'
        scaler_file = f'models/directional_{timeframe}_scaler.pkl'

        joblib.dump(model, model_file)
        joblib.dump(scaler, scaler_file)

        logger.success(f"  ✅ 모델 저장: {model_file}")

        return accuracy, high_conf_acc if high_conf_mask.any() else 0

def main():
    """메인 실행"""
    trainer = DirectionalTradingModels()

    logger.info("="*80)
    logger.info("🎯 타임프레임별 방향성 예측 모델 훈련")
    logger.info("="*80)
    logger.info("")
    logger.info("전략:")
    logger.info("  15분: 스캘핑 (단기 모멘텀)")
    logger.info("  30분: 스윙 트레이딩 (중기 트렌드)")
    logger.info("  4시간: 포지션 트레이딩 (지지/저항)")
    logger.info("  1일: 트렌드 팔로잉 (장기 방향)")
    logger.info("")

    results = {}

    # 각 타임프레임 훈련
    for timeframe in ['15m', '30m', '4h', '1d']:
        accuracy, high_conf_acc = trainer.train_model(timeframe)
        results[timeframe] = {
            'accuracy': accuracy,
            'high_conf': high_conf_acc
        }

    # 최종 결과
    logger.info("\n" + "="*80)
    logger.info("📊 최종 결과 요약")
    logger.info("="*80)

    for timeframe, result in results.items():
        strategy = trainer.strategies[timeframe]['type']
        logger.info(f"  {timeframe:4s} ({strategy:10s}): {result['accuracy']*100:5.1f}% | 고신뢰도: {result['high_conf']*100:5.1f}%")

    logger.info("")
    logger.success("✅ 모든 방향성 모델 훈련 완료!")
    logger.info("각 타임프레임별로 독립적인 거래 신호를 생성할 수 있습니다.")

if __name__ == "__main__":
    main()