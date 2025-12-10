#!/usr/bin/env python3
"""
다양한 전략으로 여러 모델을 훈련하고 최고 성능 모델 선별
전략: 모멘텀, 평균회귀, 브레이크아웃, 트렌드팔로잉, 볼륨기반
"""

import ccxt
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
import xgboost as xgb
import joblib
import os
from datetime import datetime
from loguru import logger
import warnings
warnings.filterwarnings('ignore')

class MultiStrategyTrader:
    def __init__(self):
        self.exchange = ccxt.binance()
        self.results = {}

        # 다양한 전략 정의
        self.strategies = {
            'momentum': {
                'description': '단기 모멘텀 추종',
                'features': ['rsi', 'momentum', 'rate_of_change'],
                'threshold': 0.0007,  # 0.07%
                'lookback': [3, 5, 7, 14]
            },
            'mean_reversion': {
                'description': '평균 회귀 전략',
                'features': ['bollinger', 'rsi_extreme', 'price_deviation'],
                'threshold': 0.001,  # 0.1%
                'lookback': [10, 20, 30]
            },
            'breakout': {
                'description': '레벨 돌파 전략',
                'features': ['support_resistance', 'volume_spike', 'atr'],
                'threshold': 0.0015,  # 0.15%
                'lookback': [20, 50]
            },
            'trend_following': {
                'description': '추세 추종 전략',
                'features': ['moving_averages', 'macd', 'adx'],
                'threshold': 0.002,  # 0.2%
                'lookback': [20, 50, 100]
            },
            'volume_based': {
                'description': '볼륨 기반 전략',
                'features': ['obv', 'volume_ratio', 'cvd'],
                'threshold': 0.001,
                'lookback': [5, 10, 20]
            },
            'pattern_recognition': {
                'description': '캔들 패턴 인식',
                'features': ['candle_patterns', 'price_action'],
                'threshold': 0.0008,
                'lookback': [3, 5, 10]
            },
            'volatility': {
                'description': '변동성 기반',
                'features': ['atr', 'volatility_ratio', 'bb_width'],
                'threshold': 0.0012,
                'lookback': [10, 20, 30]
            },
            'sentiment': {
                'description': '시장 심리 지표',
                'features': ['fear_greed', 'put_call_ratio', 'vix_proxy'],
                'threshold': 0.001,
                'lookback': [7, 14, 30]
            }
        }

    def get_data(self, timeframe='15m', limit=5000):
        """데이터 수집"""
        logger.info(f"  📊 데이터 수집: {timeframe} ({limit}개)")
        ohlcv = self.exchange.fetch_ohlcv('BTC/USDT', timeframe=timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        return df

    def create_momentum_features(self, df):
        """모멘텀 전략 특징"""
        features = pd.DataFrame(index=df.index)

        # RSI
        for period in [7, 14, 21]:
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss.replace(0, 1)
            features[f'rsi_{period}'] = 100 - (100 / (1 + rs))

        # 모멘텀
        for period in [3, 5, 10, 20]:
            features[f'momentum_{period}'] = df['close'].pct_change(period) * 100

        # ROC (Rate of Change)
        for period in [5, 10, 20]:
            features[f'roc_{period}'] = ((df['close'] - df['close'].shift(period)) / df['close'].shift(period)) * 100

        # 가속도
        features['acceleration'] = features['momentum_5'].diff()

        return features.fillna(0)

    def create_mean_reversion_features(self, df):
        """평균회귀 전략 특징"""
        features = pd.DataFrame(index=df.index)

        # 볼린저 밴드
        for period in [10, 20, 30]:
            sma = df['close'].rolling(period).mean()
            std = df['close'].rolling(period).std()
            features[f'bb_position_{period}'] = (df['close'] - sma) / (std * 2)
            features[f'bb_width_{period}'] = (std * 2) / sma * 100

        # RSI 극단값
        rsi_14 = self.calculate_rsi(df['close'], 14)
        features['rsi_oversold'] = (rsi_14 < 30).astype(int)
        features['rsi_overbought'] = (rsi_14 > 70).astype(int)

        # 가격 편차
        for period in [10, 20, 50]:
            ma = df['close'].rolling(period).mean()
            features[f'price_deviation_{period}'] = (df['close'] - ma) / ma * 100

        # Z-score
        for period in [20, 50]:
            mean = df['close'].rolling(period).mean()
            std = df['close'].rolling(period).std()
            features[f'zscore_{period}'] = (df['close'] - mean) / std

        return features.fillna(0)

    def create_breakout_features(self, df):
        """브레이크아웃 전략 특징"""
        features = pd.DataFrame(index=df.index)

        # 지지/저항 레벨
        for period in [20, 50, 100]:
            features[f'high_ratio_{period}'] = df['close'] / df['high'].rolling(period).max()
            features[f'low_ratio_{period}'] = df['close'] / df['low'].rolling(period).min()
            features[f'range_position_{period}'] = (df['close'] - df['low'].rolling(period).min()) / \
                                                   (df['high'].rolling(period).max() - df['low'].rolling(period).min())

        # 볼륨 스파이크
        features['volume_spike'] = df['volume'] / df['volume'].rolling(20).mean()
        features['volume_breakout'] = (df['volume'] > df['volume'].rolling(20).mean() * 1.5).astype(int)

        # ATR
        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift())
        low_close = abs(df['low'] - df['close'].shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)

        for period in [10, 20]:
            features[f'atr_{period}'] = true_range.rolling(period).mean() / df['close'] * 100

        # 돌파 신호
        features['new_high_20'] = (df['high'] == df['high'].rolling(20).max()).astype(int)
        features['new_low_20'] = (df['low'] == df['low'].rolling(20).min()).astype(int)

        return features.fillna(0)

    def create_trend_following_features(self, df):
        """추세추종 전략 특징"""
        features = pd.DataFrame(index=df.index)

        # 이동평균
        periods = [10, 20, 50, 100, 200]
        for period in periods:
            ma = df['close'].rolling(period).mean()
            features[f'ma_{period}_ratio'] = df['close'] / ma
            features[f'ma_{period}_slope'] = ma.pct_change(5) * 100

        # 이동평균 정렬
        ma_20 = df['close'].rolling(20).mean()
        ma_50 = df['close'].rolling(50).mean()
        ma_100 = df['close'].rolling(100).mean()
        features['ma_alignment'] = ((df['close'] > ma_20) & (ma_20 > ma_50) & (ma_50 > ma_100)).astype(int)

        # MACD
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        features['macd'] = exp1 - exp2
        features['macd_signal'] = features['macd'].ewm(span=9, adjust=False).mean()
        features['macd_histogram'] = features['macd'] - features['macd_signal']

        # ADX (Average Directional Index)
        plus_dm = df['high'].diff()
        minus_dm = -df['low'].diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0

        tr = true_range = pd.concat([
            df['high'] - df['low'],
            abs(df['high'] - df['close'].shift()),
            abs(df['low'] - df['close'].shift())
        ], axis=1).max(axis=1)

        atr_14 = tr.rolling(14).mean()
        plus_di = 100 * (plus_dm.rolling(14).mean() / atr_14)
        minus_di = 100 * (minus_dm.rolling(14).mean() / atr_14)
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        features['adx'] = dx.rolling(14).mean()

        return features.fillna(0)

    def create_volume_based_features(self, df):
        """볼륨 기반 전략 특징"""
        features = pd.DataFrame(index=df.index)

        # OBV (On Balance Volume)
        obv = (df['volume'] * np.sign(df['close'].diff())).cumsum()
        features['obv'] = obv
        features['obv_ma'] = obv / obv.rolling(20).mean()

        # 볼륨 비율
        features['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
        features['volume_trend'] = df['volume'].rolling(20).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0])

        # CVD (Cumulative Volume Delta) 근사
        features['cvd'] = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low']) * df['volume']
        features['cvd_cumsum'] = features['cvd'].cumsum()

        # VWAP
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        features['vwap'] = (typical_price * df['volume']).cumsum() / df['volume'].cumsum()
        features['price_vwap_ratio'] = df['close'] / features['vwap']

        # 볼륨 가중 모멘텀
        features['volume_momentum'] = (df['close'].pct_change() * df['volume']).rolling(10).sum()

        return features.fillna(0)

    def create_pattern_features(self, df):
        """캔들 패턴 특징"""
        features = pd.DataFrame(index=df.index)

        # 기본 캔들 정보
        features['body'] = abs(df['close'] - df['open']) / df['close'] * 100
        features['upper_shadow'] = (df['high'] - df[['close', 'open']].max(axis=1)) / df['close'] * 100
        features['lower_shadow'] = (df[['close', 'open']].min(axis=1) - df['low']) / df['close'] * 100

        # 패턴
        features['doji'] = (features['body'] < 0.1).astype(int)
        features['hammer'] = ((features['lower_shadow'] > features['body'] * 2) &
                             (features['upper_shadow'] < features['body'] * 0.5)).astype(int)
        features['shooting_star'] = ((features['upper_shadow'] > features['body'] * 2) &
                                    (features['lower_shadow'] < features['body'] * 0.5)).astype(int)

        # 연속 패턴
        features['higher_high'] = (df['high'] > df['high'].shift(1)).astype(int)
        features['lower_low'] = (df['low'] < df['low'].shift(1)).astype(int)
        features['inside_bar'] = ((df['high'] < df['high'].shift(1)) &
                                 (df['low'] > df['low'].shift(1))).astype(int)

        # 가격 액션
        features['rejection_up'] = features['upper_shadow'] / features['body']
        features['rejection_down'] = features['lower_shadow'] / features['body']

        return features.fillna(0)

    def calculate_rsi(self, series, period=14):
        """RSI 계산 헬퍼"""
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss.replace(0, 1)
        return 100 - (100 / (1 + rs))

    def create_labels(self, df, threshold):
        """라벨 생성"""
        future_return = df['close'].shift(-1) / df['close'] - 1
        return (future_return > threshold).astype(int)

    def train_strategy_models(self, strategy_name, timeframe='15m'):
        """각 전략별로 여러 모델 훈련"""
        logger.info(f"\n{'='*60}")
        logger.info(f"🎯 {strategy_name.upper()} 전략 모델 훈련")
        logger.info(f"   설명: {self.strategies[strategy_name]['description']}")
        logger.info(f"{'='*60}")

        # 데이터 수집
        df = self.get_data(timeframe, limit=5000)

        # 전략별 특징 생성
        if strategy_name == 'momentum':
            features = self.create_momentum_features(df)
        elif strategy_name == 'mean_reversion':
            features = self.create_mean_reversion_features(df)
        elif strategy_name == 'breakout':
            features = self.create_breakout_features(df)
        elif strategy_name == 'trend_following':
            features = self.create_trend_following_features(df)
        elif strategy_name == 'volume_based':
            features = self.create_volume_based_features(df)
        elif strategy_name == 'pattern_recognition':
            features = self.create_pattern_features(df)
        else:
            # 변동성, 심리 등 추가 전략
            features = self.create_momentum_features(df)  # 기본값

        # 라벨 생성
        labels = self.create_labels(df, self.strategies[strategy_name]['threshold'])

        # 데이터 준비
        X = features.dropna()
        y = labels[X.index][:-1]
        X = X[:-1]

        # 훈련/테스트 분할
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        # 클래스 가중치
        if len(np.unique(y_train)) > 1:
            class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
            class_weight_dict = dict(zip(np.unique(y_train), class_weights))
        else:
            class_weight_dict = {0: 1.0, 1: 1.0}

        # 스케일링
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # 여러 모델 테스트
        models = {
            'RandomForest': RandomForestClassifier(
                n_estimators=100, max_depth=10,
                class_weight=class_weight_dict, random_state=42, n_jobs=-1
            ),
            'GradientBoost': GradientBoostingClassifier(
                n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42
            ),
            'XGBoost': xgb.XGBClassifier(
                n_estimators=100, learning_rate=0.1, max_depth=5,
                scale_pos_weight=len(y_train[y_train==0])/len(y_train[y_train==1]),
                random_state=42, use_label_encoder=False, eval_metric='logloss'
            ),
            'AdaBoost': AdaBoostClassifier(
                n_estimators=50, learning_rate=1.0, random_state=42
            ),
            'NeuralNet': MLPClassifier(
                hidden_layer_sizes=(50, 30), max_iter=500,
                early_stopping=True, random_state=42
            )
        }

        strategy_results = {}

        for model_name, model in models.items():
            try:
                # 훈련
                model.fit(X_train_scaled, y_train)

                # 예측
                y_pred = model.predict(X_test_scaled)
                y_pred_proba = model.predict_proba(X_test_scaled)

                # 평가
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, zero_division=0)
                recall = recall_score(y_test, y_pred, zero_division=0)
                f1 = f1_score(y_test, y_pred, zero_division=0)

                # 고신뢰도 정확도
                high_conf_mask = np.max(y_pred_proba, axis=1) >= 0.65
                if high_conf_mask.any():
                    high_conf_acc = accuracy_score(y_test[high_conf_mask], y_pred[high_conf_mask])
                    high_conf_ratio = high_conf_mask.sum() / len(y_test)
                else:
                    high_conf_acc = 0
                    high_conf_ratio = 0

                # 결과 저장
                result = {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'high_conf_acc': high_conf_acc,
                    'high_conf_ratio': high_conf_ratio,
                    'model': model,
                    'scaler': scaler
                }

                strategy_results[model_name] = result

                logger.info(f"  {model_name:12s}: 정확도 {accuracy*100:5.1f}% | 고신뢰도 {high_conf_acc*100:5.1f}% ({high_conf_ratio*100:4.1f}%)")

            except Exception as e:
                logger.error(f"  {model_name} 오류: {str(e)}")

        # 최고 성능 모델 선택
        best_model_name = max(strategy_results.keys(),
                             key=lambda x: strategy_results[x]['high_conf_acc'] * strategy_results[x]['high_conf_ratio'])
        best_result = strategy_results[best_model_name]

        logger.success(f"  🏆 최고 모델: {best_model_name} (고신뢰도 {best_result['high_conf_acc']*100:.1f}%)")

        # 모델 저장
        model_file = f'models/{strategy_name}_{timeframe}_{best_model_name.lower()}_model.pkl'
        scaler_file = f'models/{strategy_name}_{timeframe}_{best_model_name.lower()}_scaler.pkl'

        joblib.dump(best_result['model'], model_file)
        joblib.dump(best_result['scaler'], scaler_file)

        return strategy_name, best_model_name, best_result

    def compare_all_strategies(self, timeframe='15m'):
        """모든 전략 비교"""
        all_results = []

        for strategy_name in self.strategies.keys():
            try:
                strategy, model_name, result = self.train_strategy_models(strategy_name, timeframe)
                all_results.append({
                    'strategy': strategy,
                    'model': model_name,
                    'accuracy': result['accuracy'],
                    'high_conf_acc': result['high_conf_acc'],
                    'high_conf_ratio': result['high_conf_ratio'],
                    'f1': result['f1']
                })
            except Exception as e:
                logger.error(f"전략 {strategy_name} 실패: {str(e)}")

        return all_results

def main():
    """메인 실행"""
    trainer = MultiStrategyTrader()

    logger.info("="*80)
    logger.info("🚀 다양한 전략 모델 훈련 및 비교")
    logger.info("="*80)

    # 각 타임프레임별로 테스트
    timeframes = ['15m', '30m', '1h', '4h']
    final_results = {}

    for timeframe in timeframes:
        logger.info(f"\n\n{'#'*80}")
        logger.info(f"📈 {timeframe} 타임프레임 전략 테스트")
        logger.info(f"{'#'*80}")

        results = trainer.compare_all_strategies(timeframe)
        final_results[timeframe] = results

    # 최종 결과 출력
    logger.info("\n\n" + "="*80)
    logger.info("📊 최종 성능 비교")
    logger.info("="*80)

    for timeframe, results in final_results.items():
        logger.info(f"\n📈 {timeframe} 타임프레임:")

        # 성능순 정렬
        sorted_results = sorted(results, key=lambda x: x['high_conf_acc'] * x['high_conf_ratio'], reverse=True)

        for i, result in enumerate(sorted_results[:3], 1):  # 상위 3개만
            logger.info(f"  {i}. {result['strategy']:20s} ({result['model']:12s}): "
                       f"정확도 {result['accuracy']*100:5.1f}% | "
                       f"고신뢰도 {result['high_conf_acc']*100:5.1f}% ({result['high_conf_ratio']*100:4.1f}% 신호)")

    # 최고 성능 모델들 선별
    logger.info("\n" + "="*80)
    logger.info("🏆 선별된 최고 성능 모델")
    logger.info("="*80)

    # 사용할 모델 정리
    best_models = []
    for timeframe, results in final_results.items():
        if results:
            best = max(results, key=lambda x: x['high_conf_acc'] * x['high_conf_ratio'])
            if best['high_conf_acc'] >= 0.6:  # 60% 이상만 사용
                best_models.append(f"{best['strategy']}_{timeframe}_{best['model'].lower()}")
                logger.success(f"✅ {timeframe}: {best['strategy']} ({best['model']}) - {best['high_conf_acc']*100:.1f}%")

    # 모델 파일 정리
    logger.info("\n📁 모델 파일 정리 중...")
    os.makedirs('../models', exist_ok=True)

    # models 폴더의 모든 파일 확인
    model_files = [f for f in os.listdir('models') if f.endswith('.pkl')]

    for file in model_files:
        # 파일명에서 모델 정보 추출
        base_name = file.replace('_model.pkl', '').replace('_scaler.pkl', '')

        # 사용할 모델인지 확인
        is_best = any(base_name.startswith(model) for model in best_models)

        if not is_best:
            # 사용하지 않는 모델은 ../models로 이동
            os.rename(f'models/{file}', f'../models/{file}')
            logger.info(f"  📦 {file} → ../models/")

    logger.success("\n✅ 완료! 최고 성능 모델만 btc_trading_system/models에 보관")
    logger.info("나머지 모델은 ../models로 이동했습니다.")

if __name__ == "__main__":
    main()