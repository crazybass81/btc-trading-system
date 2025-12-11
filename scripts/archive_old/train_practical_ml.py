#!/usr/bin/env python3
"""
실용적 ML 모델 학습
현실적인 목표: 55-60% 정확도
"""

import pandas as pd
import numpy as np
from datetime import datetime
import ccxt
import joblib
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from loguru import logger
import warnings
warnings.filterwarnings('ignore')

class PracticalMLTrainer:
    def __init__(self):
        self.exchange = ccxt.binance()
        self.models = {}
        self.scalers = {}

    def prepare_features(self, df):
        """간단하지만 효과적인 특징 생성"""
        features = pd.DataFrame()

        # 가격 변화율
        for i in [1, 3, 5, 10]:
            features[f'return_{i}'] = df['close'].pct_change(i)

        # RSI
        for period in [7, 14, 21]:
            features[f'rsi_{period}'] = self.calculate_rsi(df['close'], period)

        # MACD
        exp1 = df['close'].ewm(span=12).mean()
        exp2 = df['close'].ewm(span=26).mean()
        features['macd'] = exp1 - exp2
        features['macd_signal'] = features['macd'].ewm(span=9).mean()
        features['macd_hist'] = features['macd'] - features['macd_signal']

        # 볼린저 밴드
        for period in [10, 20]:
            sma = df['close'].rolling(period).mean()
            std = df['close'].rolling(period).std()
            features[f'bb_position_{period}'] = (df['close'] - sma) / (2 * std)

        # 볼륨
        features['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
        features['volume_change'] = df['volume'].pct_change()

        # 고저 범위
        features['high_low_ratio'] = (df['high'] - df['low']) / df['close']
        features['close_position'] = (df['close'] - df['low']) / (df['high'] - df['low'])

        return features

    def calculate_rsi(self, prices, period=14):
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def create_labels(self, df, threshold=0.002):
        """3클래스 레이블: 상승/하락/중립"""
        future_return = df['close'].shift(-1) / df['close'] - 1
        labels = pd.Series(1, index=df.index)  # 기본값 중립

        labels[future_return > threshold] = 2  # 상승
        labels[future_return < -threshold] = 0  # 하락

        return labels

    def train_model(self, timeframe='15m'):
        """단일 타임프레임 모델 학습"""
        logger.info(f"\n{'='*50}")
        logger.info(f"🚀 {timeframe} 모델 학습 시작")
        logger.info(f"{'='*50}")

        # 데이터 수집
        logger.info("데이터 수집 중...")
        ohlcv = self.exchange.fetch_ohlcv('BTC/USDT', timeframe, limit=1000)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

        # 특징 생성
        logger.info("특징 생성 중...")
        features = self.prepare_features(df)

        # 레이블 생성
        labels = self.create_labels(df)

        # 데이터 정리
        X = features.dropna()
        y = labels[X.index]
        X = X[:-1]  # 마지막 행 제거 (미래 레이블 없음)
        y = y[:-1]

        logger.info(f"데이터 크기: {X.shape}")
        logger.info(f"클래스 분포: {y.value_counts().to_dict()}")

        # 학습/테스트 분할
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=False
        )

        # 스케일링
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # 모델 학습
        logger.info("모델 학습 중...")

        # 1. Random Forest
        rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=10,
            random_state=42
        )
        rf.fit(X_train_scaled, y_train)
        rf_pred = rf.predict(X_test_scaled)
        rf_acc = accuracy_score(y_test, rf_pred)

        # 2. Gradient Boosting
        gb = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
        gb.fit(X_train_scaled, y_train)
        gb_pred = gb.predict(X_test_scaled)
        gb_acc = accuracy_score(y_test, gb_pred)

        # 앙상블 (투표)
        ensemble_pred = np.round((rf_pred + gb_pred) / 2).astype(int)
        ensemble_acc = accuracy_score(y_test, ensemble_pred)

        logger.info(f"\n📊 학습 결과:")
        logger.info(f"Random Forest 정확도: {rf_acc:.1%}")
        logger.info(f"Gradient Boosting 정확도: {gb_acc:.1%}")
        logger.info(f"앙상블 정확도: {ensemble_acc:.1%}")

        # 최고 모델 선택
        if ensemble_acc >= max(rf_acc, gb_acc):
            self.models[timeframe] = {'rf': rf, 'gb': gb, 'type': 'ensemble'}
            best_acc = ensemble_acc
            logger.success(f"✅ 앙상블 모델 선택 (정확도: {ensemble_acc:.1%})")
        elif rf_acc > gb_acc:
            self.models[timeframe] = {'model': rf, 'type': 'rf'}
            best_acc = rf_acc
            logger.success(f"✅ Random Forest 선택 (정확도: {rf_acc:.1%})")
        else:
            self.models[timeframe] = {'model': gb, 'type': 'gb'}
            best_acc = gb_acc
            logger.success(f"✅ Gradient Boosting 선택 (정확도: {gb_acc:.1%})")

        self.scalers[timeframe] = scaler

        # 상세 리포트
        if best_acc > 0.55:
            logger.info(f"\n🎯 목표 달성! (55% 이상)")
        else:
            logger.warning(f"\n⚠️ 목표 미달 (현재: {best_acc:.1%}, 목표: 55%)")

        return best_acc

    def save_models(self):
        """모델 저장"""
        for timeframe, model_dict in self.models.items():
            model_path = f'models/practical_{timeframe}_model.pkl'
            scaler_path = f'models/practical_{timeframe}_scaler.pkl'

            joblib.dump(model_dict, model_path)
            joblib.dump(self.scalers[timeframe], scaler_path)

            logger.info(f"✅ {timeframe} 모델 저장 완료: {model_path}")

    def train_all_timeframes(self):
        """모든 타임프레임 학습"""
        results = {}

        for timeframe in ['5m', '15m', '1h', '4h']:
            try:
                accuracy = self.train_model(timeframe)
                results[timeframe] = accuracy
            except Exception as e:
                logger.error(f"{timeframe} 학습 실패: {e}")
                results[timeframe] = 0

        return results

def main():
    trainer = PracticalMLTrainer()

    logger.info("="*70)
    logger.info("🤖 실용적 ML 모델 학습")
    logger.info("목표: 55-60% 정확도 (현실적 목표)")
    logger.info("="*70)

    # 모든 타임프레임 학습
    results = trainer.train_all_timeframes()

    # 결과 요약
    logger.info("\n" + "="*70)
    logger.info("📋 학습 결과 요약")
    logger.info("="*70)

    total_models = len(results)
    successful_models = sum(1 for acc in results.values() if acc > 0.55)

    for timeframe, accuracy in results.items():
        status = "✅" if accuracy > 0.55 else "❌"
        logger.info(f"{timeframe}: {accuracy:.1%} {status}")

    logger.info(f"\n성공률: {successful_models}/{total_models}")

    # 모델 저장
    if successful_models > 0:
        trainer.save_models()
        logger.success(f"\n✅ {successful_models}개 모델 저장 완료")

    # 현실적 조언
    logger.info("\n" + "="*70)
    logger.info("💡 사용자님께")
    logger.info("="*70)
    logger.info("\n이 ML 모델의 현실:")
    logger.info("1. 정확도는 55-60% 수준이 한계")
    logger.info("2. BTC는 본질적으로 예측 어려움")
    logger.info("3. 과도한 신뢰는 위험")
    logger.info("4. 리스크 관리와 함께 사용 필수")
    logger.info("5. 보조 지표로만 활용 권장")

if __name__ == "__main__":
    main()