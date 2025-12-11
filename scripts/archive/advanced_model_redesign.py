#!/usr/bin/env python3
"""
고급 모델 재설계 및 검증
1. 성공 모델 (15분, 1시간) 추가 검증
2. 30분 모델 신규 개발
3. 실패 모델 (4시간, 1일) 대안 접근법
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import ccxt
import joblib
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.feature_selection import SelectKBest, f_classif
from loguru import logger
import warnings
warnings.filterwarnings('ignore')

class AdvancedModelSystem:
    def __init__(self):
        self.exchange = ccxt.binance()
        self.models = {}
        self.scalers = {}

    def enhanced_features(self, df, timeframe):
        """향상된 특징 생성"""
        features = pd.DataFrame(index=df.index)

        # 기본 가격 특징
        for period in [1, 2, 3, 5, 7, 10, 15, 20]:
            if len(df) > period:
                features[f'return_{period}'] = df['close'].pct_change(period)
                features[f'volume_change_{period}'] = df['volume'].pct_change(period)

        # RSI 다중 기간
        for period in [7, 14, 21, 28]:
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
            rs = gain / loss
            features[f'rsi_{period}'] = 100 - (100 / (1 + rs))

        # MACD 변형
        for fast, slow in [(12, 26), (5, 35), (10, 20)]:
            exp1 = df['close'].ewm(span=fast).mean()
            exp2 = df['close'].ewm(span=slow).mean()
            features[f'macd_{fast}_{slow}'] = exp1 - exp2

        # 볼린저 밴드 다중
        for period in [10, 20, 30]:
            sma = df['close'].rolling(period).mean()
            std = df['close'].rolling(period).std()
            features[f'bb_width_{period}'] = (2 * std) / sma
            features[f'bb_position_{period}'] = (df['close'] - sma) / (2 * std)

        # 볼륨 프로파일
        features['volume_sma_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
        features['volume_std'] = df['volume'].rolling(20).std() / df['volume'].rolling(20).mean()

        # 변동성 지표
        features['true_range'] = pd.concat([
            df['high'] - df['low'],
            abs(df['high'] - df['close'].shift()),
            abs(df['low'] - df['close'].shift())
        ], axis=1).max(axis=1)
        features['atr'] = features['true_range'].rolling(14).mean() / df['close']

        # 패턴 인식
        features['doji'] = (abs(df['close'] - df['open']) / (df['high'] - df['low'])).rolling(3).mean()
        features['pin_bar'] = ((df['high'] - df['close']) / (df['high'] - df['low'])).rolling(3).mean()

        # 타임프레임별 특수 특징
        if timeframe in ['30m', '1h']:
            # 중기 트렌드
            for period in [50, 100]:
                if len(df) > period:
                    features[f'ma_{period}_slope'] = df['close'].rolling(period).mean().pct_change(5)

        elif timeframe in ['4h', '1d']:
            # 장기 트렌드 (다른 접근)
            if len(df) > 200:
                features['long_trend'] = df['close'] / df['close'].rolling(200).mean()
                features['trend_strength'] = abs(features['long_trend'] - 1)

        return features

    def verify_successful_models(self):
        """성공한 모델 (15분, 1시간) 추가 검증"""
        logger.info("="*70)
        logger.info("🔍 성공 모델 추가 검증")
        logger.info("="*70)

        results = {}

        for timeframe in ['15m', '1h']:
            logger.info(f"\n{timeframe} 모델 심화 검증...")

            # 모델 로드
            try:
                model_path = f'models/practical_{timeframe}_model.pkl'
                scaler_path = f'models/practical_{timeframe}_scaler.pkl'

                model_dict = joblib.load(model_path)
                scaler = joblib.load(scaler_path)

                if 'model' in model_dict:
                    model = model_dict['model']
                else:
                    model = model_dict

                logger.info(f"✅ {timeframe} 모델 로드 성공")
            except:
                logger.warning(f"⚠️ {timeframe} 모델 로드 실패")
                continue

            # 다양한 기간으로 검증
            test_periods = {
                '3일': 288 if timeframe == '15m' else 72,
                '7일': 672 if timeframe == '15m' else 168,
                '14일': 1344 if timeframe == '15m' else 336
            }

            period_results = {}

            for period_name, limit in test_periods.items():
                ohlcv = self.exchange.fetch_ohlcv('BTC/USDT', timeframe, limit=limit)
                df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

                # 특징 생성
                features = self.prepare_basic_features(df)
                labels = self.create_labels(df)

                X = features.dropna()
                y = labels[X.index][:-1]
                X = X[:-1]

                # 예측 및 평가
                correct = 0
                high_conf_correct = 0
                high_conf_total = 0

                for i in range(len(X) // 2, len(X)):
                    X_test = X.iloc[i:i+1]
                    y_true = y.iloc[i]

                    X_scaled = scaler.transform(X_test)

                    if hasattr(model, 'predict_proba'):
                        y_pred = model.predict(X_scaled)[0]
                        proba = model.predict_proba(X_scaled)[0]
                        confidence = max(proba) * 100

                        if confidence >= 70:
                            high_conf_total += 1
                            if y_pred == y_true:
                                high_conf_correct += 1
                    else:
                        y_pred = model.predict(X_scaled)[0]
                        confidence = 60  # 기본값

                    if y_pred == y_true:
                        correct += 1

                accuracy = (correct / (len(X) - len(X) // 2)) * 100
                high_conf_acc = (high_conf_correct / high_conf_total * 100) if high_conf_total > 0 else 0

                period_results[period_name] = {
                    'accuracy': accuracy,
                    'high_conf_accuracy': high_conf_acc,
                    'high_conf_count': high_conf_total
                }

                logger.info(f"  {period_name}: {accuracy:.1f}% (고신뢰도: {high_conf_acc:.1f}%)")

            results[timeframe] = period_results

        return results

    def develop_30m_model(self):
        """30분 모델 신규 개발"""
        logger.info("\n" + "="*70)
        logger.info("🚀 30분 모델 신규 개발")
        logger.info("="*70)

        # 데이터 수집
        logger.info("데이터 수집 중...")
        ohlcv = self.exchange.fetch_ohlcv('BTC/USDT', '30m', limit=1000)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

        # 향상된 특징 생성
        logger.info("향상된 특징 생성 중...")
        features = self.enhanced_features(df, '30m')
        labels = self.create_labels(df, threshold=0.0025)  # 30분용 임계값

        # 데이터 정리
        X = features.dropna()
        y = labels[X.index][:-1]
        X = X[:-1]

        logger.info(f"데이터 크기: {X.shape}")
        logger.info(f"클래스 분포: {y.value_counts().to_dict()}")

        # 특징 선택 (가장 중요한 30개)
        selector = SelectKBest(f_classif, k=min(30, X.shape[1]))
        X_selected = selector.fit_transform(X, y)
        selected_features = X.columns[selector.get_support()]
        logger.info(f"선택된 특징 수: {len(selected_features)}")

        # 학습/테스트 분할
        X_train, X_test, y_train, y_test = train_test_split(
            X_selected, y, test_size=0.2, shuffle=False
        )

        # 스케일링
        scaler = RobustScaler()  # 이상치에 강한 스케일러
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # 다양한 모델 테스트
        models = {
            'RandomForest': RandomForestClassifier(
                n_estimators=200, max_depth=10, min_samples_split=10, random_state=42
            ),
            'GradientBoosting': GradientBoostingClassifier(
                n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42
            ),
            'AdaBoost': AdaBoostClassifier(
                n_estimators=100, learning_rate=1.0, random_state=42
            )
        }

        best_score = 0
        best_model = None
        best_name = None

        for name, model in models.items():
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')

            logger.info(f"\n{name}:")
            logger.info(f"  정확도: {accuracy:.1%}")
            logger.info(f"  정밀도: {precision:.1%}")
            logger.info(f"  재현율: {recall:.1%}")
            logger.info(f"  F1: {f1:.1%}")

            if accuracy > best_score:
                best_score = accuracy
                best_model = model
                best_name = name

        logger.success(f"\n✅ 최고 모델: {best_name} (정확도: {best_score:.1%})")

        # 모델 저장
        if best_score > 0.55:
            self.models['30m'] = best_model
            self.scalers['30m'] = scaler

            joblib.dump(best_model, 'models/advanced_30m_model.pkl')
            joblib.dump(scaler, 'models/advanced_30m_scaler.pkl')
            joblib.dump(selected_features, 'models/advanced_30m_features.pkl')

            logger.success("✅ 30분 모델 저장 완료")

        return best_score

    def redesign_long_term_models(self):
        """장기 모델 대안 접근법 - 트렌드 분류로 변경"""
        logger.info("\n" + "="*70)
        logger.info("🔄 장기 모델 대안 접근법")
        logger.info("="*70)

        results = {}

        for timeframe in ['4h', '1d']:
            logger.info(f"\n{timeframe} 대안 모델 개발...")

            # 데이터 수집
            limit = 1000 if timeframe == '4h' else 365
            ohlcv = self.exchange.fetch_ohlcv('BTC/USDT', timeframe, limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

            # 트렌드 레이블 생성 (단순 상승/하락 대신 트렌드 강도)
            labels = self.create_trend_labels(df, timeframe)

            # 특징 생성 - 트렌드 중심
            features = self.create_trend_features(df, timeframe)

            # 데이터 정리
            X = features.dropna()
            y = labels[X.index][:-1]
            X = X[:-1]

            logger.info(f"데이터 크기: {X.shape}")
            logger.info(f"트렌드 분포: {y.value_counts().to_dict()}")

            # 학습/테스트 분할
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, shuffle=False
            )

            # 스케일링
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # 간단한 트렌드 분류기
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=5,  # 얕은 트리로 과적합 방지
                min_samples_split=20,
                random_state=42
            )

            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)

            logger.info(f"{timeframe} 트렌드 분류 정확도: {accuracy:.1%}")

            if accuracy > 0.5:  # 50% 이상이면 저장
                joblib.dump(model, f'models/trend_{timeframe}_model.pkl')
                joblib.dump(scaler, f'models/trend_{timeframe}_scaler.pkl')
                logger.success(f"✅ {timeframe} 트렌드 모델 저장")

            results[timeframe] = accuracy

        return results

    def create_trend_labels(self, df, timeframe):
        """트렌드 강도 레이블 (상승/횡보/하락)"""
        if timeframe == '4h':
            lookback = 10  # 40시간
            threshold = 0.02  # 2%
        else:  # 1d
            lookback = 7  # 7일
            threshold = 0.03  # 3%

        trend = (df['close'] / df['close'].shift(lookback) - 1)

        labels = pd.Series(1, index=df.index)  # 기본 횡보
        labels[trend > threshold] = 2  # 강한 상승
        labels[trend < -threshold] = 0  # 강한 하락

        return labels

    def create_trend_features(self, df, timeframe):
        """트렌드 중심 특징"""
        features = pd.DataFrame(index=df.index)

        # 다양한 기간의 이동평균
        for period in [20, 50, 100, 200]:
            if len(df) > period:
                ma = df['close'].rolling(period).mean()
                features[f'ma_{period}_ratio'] = df['close'] / ma
                features[f'ma_{period}_slope'] = ma.pct_change(5)

        # 트렌드 강도
        features['trend_7d'] = df['close'].pct_change(7 if timeframe == '1d' else 42)
        features['trend_14d'] = df['close'].pct_change(14 if timeframe == '1d' else 84)
        features['trend_30d'] = df['close'].pct_change(30 if timeframe == '1d' else 180)

        # 변동성
        features['volatility_7d'] = df['close'].pct_change().rolling(7).std()
        features['volatility_30d'] = df['close'].pct_change().rolling(30).std()

        # 볼륨 트렌드
        features['volume_trend'] = df['volume'].rolling(20).mean() / df['volume'].rolling(50).mean()

        # 고저 범위
        features['high_low_range'] = (df['high'] - df['low']) / df['close']
        features['range_expansion'] = features['high_low_range'].rolling(10).mean()

        return features

    def prepare_basic_features(self, df):
        """기본 특징 (검증용)"""
        features = pd.DataFrame(index=df.index)

        for i in [1, 3, 5, 10]:
            features[f'return_{i}'] = df['close'].pct_change(i)

        for period in [7, 14, 21]:
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
            rs = gain / loss
            features[f'rsi_{period}'] = 100 - (100 / (1 + rs))

        exp1 = df['close'].ewm(span=12).mean()
        exp2 = df['close'].ewm(span=26).mean()
        features['macd'] = exp1 - exp2
        features['macd_signal'] = features['macd'].ewm(span=9).mean()
        features['macd_hist'] = features['macd'] - features['macd_signal']

        for period in [10, 20]:
            sma = df['close'].rolling(period).mean()
            std = df['close'].rolling(period).std()
            features[f'bb_position_{period}'] = (df['close'] - sma) / (2 * std)

        features['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
        features['volume_change'] = df['volume'].pct_change()
        features['high_low_ratio'] = (df['high'] - df['low']) / df['close']
        features['close_position'] = (df['close'] - df['low']) / (df['high'] - df['low'])

        return features

    def create_labels(self, df, threshold=0.002):
        """레이블 생성"""
        future_return = df['close'].shift(-1) / df['close'] - 1
        labels = pd.Series(1, index=df.index)
        labels[future_return > threshold] = 2
        labels[future_return < -threshold] = 0
        return labels

def main():
    system = AdvancedModelSystem()

    logger.info("="*70)
    logger.info("🚀 고급 모델 재설계 및 검증")
    logger.info("="*70)

    # 1. 성공 모델 추가 검증
    verification_results = system.verify_successful_models()

    # 2. 30분 모델 개발
    accuracy_30m = system.develop_30m_model()

    # 3. 장기 모델 대안 접근
    long_term_results = system.redesign_long_term_models()

    # 최종 보고
    logger.info("\n" + "="*70)
    logger.info("📊 최종 결과")
    logger.info("="*70)

    logger.info("\n✅ 검증된 모델:")
    for timeframe, periods in verification_results.items():
        logger.info(f"\n{timeframe}:")
        for period, result in periods.items():
            logger.info(f"  {period}: {result['accuracy']:.1f}%")

    logger.info(f"\n🆕 30분 모델: {accuracy_30m:.1%}")

    logger.info("\n🔄 대안 접근 (트렌드 분류):")
    for timeframe, acc in long_term_results.items():
        logger.info(f"  {timeframe}: {acc:.1%}")

    # TodoWrite 업데이트
    from datetime import datetime
    logger.info(f"\n✅ 작업 완료: {datetime.now()}")

if __name__ == "__main__":
    main()