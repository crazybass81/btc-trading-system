#!/usr/bin/env python3
"""
NEUTRAL 편향 문제를 해결하기 위한 모델 재훈련
- 타임프레임별 적절한 임계값 사용
- 클래스 균형 개선
- 실제 거래 가능한 신호 생성
"""

import ccxt
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils.class_weight import compute_class_weight
import joblib
from datetime import datetime
from loguru import logger
import warnings
warnings.filterwarnings('ignore')

class ImprovedTradingModels:
    def __init__(self):
        self.exchange = ccxt.binance()

        # 타임프레임별 적절한 임계값
        self.thresholds = {
            '15m': 0.001,  # 0.1% (기존 0.2% → 0.1%)
            '30m': 0.0015,  # 0.15%
            '4h': 0.003,   # 0.3%
            '1d': 0.005    # 0.5%
        }

        self.data_limits = {
            '15m': 3000,
            '30m': 2000,
            '4h': 1000,
            '1d': 500
        }

    def create_balanced_labels(self, df, timeframe):
        """균형잡힌 라벨 생성"""
        threshold = self.thresholds[timeframe]

        # 미래 수익률 계산
        future_return = df['close'].shift(-1) / df['close'] - 1

        # 라벨 생성
        labels = pd.Series(1, index=df.index)  # 기본값 NEUTRAL
        labels[future_return > threshold] = 2   # LONG
        labels[future_return < -threshold] = 0  # SHORT

        # 라벨 분포 출력
        label_counts = labels.value_counts().sort_index()
        total = len(labels.dropna())

        logger.info(f"  라벨 분포 ({timeframe}, 임계값 {threshold*100:.2f}%):")
        for label, count in label_counts.items():
            label_name = {0: 'SHORT', 1: 'NEUTRAL', 2: 'LONG'}[label]
            pct = (count / total) * 100
            logger.info(f"    {label_name}: {count:4d}개 ({pct:5.1f}%)")

        return labels

    def get_data(self, timeframe):
        """데이터 가져오기"""
        limit = self.data_limits[timeframe]

        logger.info(f"  데이터 수집 중... ({timeframe}, {limit}개 캔들)")

        ohlcv = self.exchange.fetch_ohlcv('BTC/USDT', timeframe=timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)

        return df

    def create_features(self, df):
        """향상된 특징 생성"""
        features = pd.DataFrame(index=df.index)

        # 1. 가격 변화율 (다양한 기간)
        for period in [1, 2, 3, 5, 7, 10, 15, 20]:
            features[f'return_{period}'] = df['close'].pct_change(period) * 100

        # 2. RSI (다양한 기간)
        for period in [7, 14, 21, 28]:
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss.replace(0, 1)
            features[f'rsi_{period}'] = 100 - (100 / (1 + rs))

        # 3. MACD
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        features['macd'] = exp1 - exp2
        features['macd_signal'] = features['macd'].ewm(span=9, adjust=False).mean()
        features['macd_histogram'] = features['macd'] - features['macd_signal']

        # 4. 볼린저 밴드
        for period in [10, 20, 30]:
            sma = df['close'].rolling(window=period).mean()
            std = df['close'].rolling(window=period).std()
            features[f'bb_width_{period}'] = (std * 2) / sma * 100
            features[f'bb_position_{period}'] = (df['close'] - sma) / (std * 2)

        # 5. 볼륨 지표
        features['volume_ratio'] = df['volume'] / df['volume'].rolling(window=20).mean()
        features['volume_change'] = df['volume'].pct_change() * 100

        # 6. 변동성
        features['volatility'] = df['close'].pct_change().rolling(window=20).std() * 100

        # 7. 고저 비율
        features['high_low_ratio'] = (df['high'] - df['low']) / df['close'] * 100

        # 8. 이동평균
        for period in [5, 10, 20, 50]:
            ma = df['close'].rolling(window=period).mean()
            features[f'ma_{period}_ratio'] = df['close'] / ma

        return features.fillna(0)

    def train_model(self, timeframe):
        """개선된 모델 훈련"""
        logger.info(f"\n{'='*60}")
        logger.info(f"🚀 {timeframe} 모델 훈련 시작")
        logger.info(f"{'='*60}")

        # 데이터 수집
        df = self.get_data(timeframe)

        # 특징 및 라벨 생성
        features = self.create_features(df)
        labels = self.create_balanced_labels(df, timeframe)

        # 정렬 및 정리
        X = features.dropna()
        y = labels[X.index][:-1]  # 마지막 라벨 제외 (미래 데이터 없음)
        X = X[:-1]

        # 훈련/테스트 분할 (시계열 유지)
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        # 클래스 가중치 계산
        classes = np.unique(y_train)
        class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
        class_weight_dict = dict(zip(classes, class_weights))

        logger.info(f"  클래스 가중치: {class_weight_dict}")

        # 스케일링
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # 앙상블 모델 생성
        rf = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=10,
            min_samples_leaf=5,
            class_weight=class_weight_dict,
            random_state=42,
            n_jobs=-1
        )

        gb = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=7,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42
        )

        # 앙상블
        model = VotingClassifier(
            estimators=[('rf', rf), ('gb', gb)],
            voting='soft',
            weights=[0.6, 0.4]
        )

        # 훈련
        logger.info("  모델 훈련 중...")
        model.fit(X_train_scaled, y_train)

        # 평가
        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)

        logger.info(f"\n  전체 정확도: {accuracy*100:.2f}%")

        # 예측 분포 확인
        pred_counts = pd.Series(y_pred).value_counts().sort_index()
        total_pred = len(y_pred)

        logger.info(f"\n  예측 분포:")
        for label, count in pred_counts.items():
            label_name = {0: 'SHORT', 1: 'NEUTRAL', 2: 'LONG'}[label]
            pct = (count / total_pred) * 100
            logger.info(f"    {label_name}: {count:4d}개 ({pct:5.1f}%)")

        # 높은 신뢰도 예측 분석
        y_pred_proba = model.predict_proba(X_test_scaled)
        high_conf_mask = np.max(y_pred_proba, axis=1) >= 0.7

        if high_conf_mask.any():
            high_conf_accuracy = accuracy_score(y_test[high_conf_mask], y_pred[high_conf_mask])
            high_conf_count = high_conf_mask.sum()
            high_conf_pct = (high_conf_count / len(y_pred)) * 100

            logger.info(f"\n  고신뢰도 (≥70%) 예측:")
            logger.info(f"    개수: {high_conf_count}개 ({high_conf_pct:.1f}%)")
            logger.info(f"    정확도: {high_conf_accuracy*100:.2f}%")

            # 고신뢰도 예측 분포
            high_conf_pred = y_pred[high_conf_mask]
            high_conf_counts = pd.Series(high_conf_pred).value_counts().sort_index()

            logger.info(f"    분포:")
            for label, count in high_conf_counts.items():
                label_name = {0: 'SHORT', 1: 'NEUTRAL', 2: 'LONG'}[label]
                pct = (count / len(high_conf_pred)) * 100
                logger.info(f"      {label_name}: {count:3d}개 ({pct:5.1f}%)")

        # 분류 보고서
        logger.info(f"\n  분류 보고서:")
        report = classification_report(y_test, y_pred,
                                      target_names=['SHORT', 'NEUTRAL', 'LONG'])
        for line in report.split('\n'):
            if line:
                logger.info(f"    {line}")

        # 모델 저장
        model_file = f'models/fixed_{timeframe.replace("m", "min").replace("h", "hour").replace("d", "day")}_model.pkl'
        scaler_file = f'models/fixed_{timeframe.replace("m", "min").replace("h", "hour").replace("d", "day")}_scaler.pkl'

        joblib.dump(model, model_file)
        joblib.dump(scaler, scaler_file)

        logger.success(f"  ✅ 모델 저장: {model_file}")

        return accuracy, model, scaler

    def test_realtime_predictions(self, model, scaler, timeframe):
        """실시간 예측 테스트"""
        logger.info(f"\n  실시간 예측 테스트:")

        # 최신 데이터로 테스트
        df = self.get_data(timeframe)
        features = self.create_features(df)

        # 최근 10개 예측
        recent_features = features.iloc[-10:]
        recent_scaled = scaler.transform(recent_features)

        predictions = model.predict(recent_scaled)
        probabilities = model.predict_proba(recent_scaled)

        for i in range(len(predictions)):
            pred = predictions[i]
            prob = probabilities[i]
            max_prob = max(prob) * 100

            label_name = {0: 'SHORT', 1: 'NEUTRAL', 2: 'LONG'}[pred]
            timestamp = recent_features.index[i].strftime('%m-%d %H:%M')

            logger.info(f"    {timestamp}: {label_name:8s} (신뢰도: {max_prob:.1f}%)")

def main():
    """메인 실행"""
    trainer = ImprovedTradingModels()

    logger.info("="*70)
    logger.info("🔧 NEUTRAL 편향 문제 해결을 위한 모델 재훈련")
    logger.info("="*70)
    logger.info("")
    logger.info("변경 사항:")
    logger.info("- 15분: 0.2% → 0.1% 임계값")
    logger.info("- 30분: 0.2% → 0.15% 임계값")
    logger.info("- 4시간: 0.2% → 0.3% 임계값")
    logger.info("- 1일: 0.2% → 0.5% 임계값")
    logger.info("- 클래스 균형 가중치 적용")
    logger.info("")

    results = {}

    # 각 타임프레임 모델 훈련
    for timeframe in ['15m', '30m', '4h', '1d']:
        accuracy, model, scaler = trainer.train_model(timeframe)
        trainer.test_realtime_predictions(model, scaler, timeframe)
        results[timeframe] = accuracy

    # 최종 결과
    logger.info("\n" + "="*70)
    logger.info("📊 최종 결과")
    logger.info("="*70)

    for timeframe, accuracy in results.items():
        logger.info(f"  {timeframe:4s}: {accuracy*100:.2f}% 정확도")

    logger.success("\n✅ 모든 모델 재훈련 완료!")
    logger.info("새로운 모델들이 models/fixed_*.pkl로 저장되었습니다.")
    logger.info("이제 실제 거래 신호를 생성할 수 있습니다!")

if __name__ == "__main__":
    main()