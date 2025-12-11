#!/usr/bin/env python3
"""
개선된 ML 모델 훈련 스크립트
- 더 나은 특징 공학
- 적응적 임계값
- 앙상블 방법
- 교차 검증
"""

import ccxt
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from xgboost import XGBClassifier
import joblib
from datetime import datetime
import talib
import warnings
warnings.filterwarnings('ignore')

class ImprovedModelTrainer:
    def __init__(self):
        self.exchange = ccxt.binance()

    def get_data(self, timeframe, limit=10000):
        """데이터 수집"""
        print(f"📊 데이터 수집: {timeframe} ({limit}개 캔들)")

        ohlcv = self.exchange.fetch_ohlcv('BTC/USDT', timeframe=timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)

        return df

    def create_advanced_features(self, df):
        """고급 특징 생성 (TA-Lib 활용)"""
        features = pd.DataFrame(index=df.index)

        # 가격 데이터 numpy 배열로 변환
        open_price = df['open'].values
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values
        volume = df['volume'].values

        # 1. 모멘텀 지표
        features['rsi'] = talib.RSI(close, timeperiod=14)
        features['rsi_fast'] = talib.RSI(close, timeperiod=7)
        features['rsi_slow'] = talib.RSI(close, timeperiod=21)

        # 2. 이동평균
        features['sma_10'] = talib.SMA(close, timeperiod=10)
        features['sma_20'] = talib.SMA(close, timeperiod=20)
        features['sma_50'] = talib.SMA(close, timeperiod=50)
        features['ema_12'] = talib.EMA(close, timeperiod=12)
        features['ema_26'] = talib.EMA(close, timeperiod=26)

        # 이동평균 비율
        features['price_sma10_ratio'] = close / features['sma_10']
        features['price_sma20_ratio'] = close / features['sma_20']
        features['sma10_sma20_ratio'] = features['sma_10'] / features['sma_20']

        # 3. MACD
        macd, macd_signal, macd_hist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
        features['macd'] = macd
        features['macd_signal'] = macd_signal
        features['macd_hist'] = macd_hist

        # 4. 볼린저 밴드
        bb_upper, bb_middle, bb_lower = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
        features['bb_upper'] = bb_upper
        features['bb_middle'] = bb_middle
        features['bb_lower'] = bb_lower
        features['bb_width'] = bb_upper - bb_lower
        features['bb_position'] = (close - bb_lower) / (bb_upper - bb_lower)

        # 5. 스토캐스틱
        slowk, slowd = talib.STOCH(high, low, close, fastk_period=14, slowk_period=3, slowd_period=3)
        features['stoch_k'] = slowk
        features['stoch_d'] = slowd

        # 6. ADX (트렌드 강도)
        features['adx'] = talib.ADX(high, low, close, timeperiod=14)
        features['plus_di'] = talib.PLUS_DI(high, low, close, timeperiod=14)
        features['minus_di'] = talib.MINUS_DI(high, low, close, timeperiod=14)

        # 7. ATR (변동성)
        features['atr'] = talib.ATR(high, low, close, timeperiod=14)
        features['atr_ratio'] = features['atr'] / close

        # 8. 볼륨 지표
        features['obv'] = talib.OBV(close, volume)
        features['ad'] = talib.AD(high, low, close, volume)
        features['adosc'] = talib.ADOSC(high, low, close, volume, fastperiod=3, slowperiod=10)

        # 9. 패턴 인식 (캔들 패턴)
        features['hammer'] = talib.CDLHAMMER(open_price, high, low, close)
        features['doji'] = talib.CDLDOJI(open_price, high, low, close)
        features['engulfing'] = talib.CDLENGULFING(open_price, high, low, close)
        features['morning_star'] = talib.CDLMORNINGSTAR(open_price, high, low, close)
        features['three_white_soldiers'] = talib.CDL3WHITESOLDIERS(open_price, high, low, close)

        # 10. 추가 지표
        features['cci'] = talib.CCI(high, low, close, timeperiod=14)
        features['mfi'] = talib.MFI(high, low, close, volume, timeperiod=14)
        features['willr'] = talib.WILLR(high, low, close, timeperiod=14)
        features['roc'] = talib.ROC(close, timeperiod=10)
        features['mom'] = talib.MOM(close, timeperiod=10)

        # 11. 가격 변화율
        features['return_1'] = df['close'].pct_change(1)
        features['return_3'] = df['close'].pct_change(3)
        features['return_5'] = df['close'].pct_change(5)
        features['return_10'] = df['close'].pct_change(10)

        # 12. 고/저 비율
        features['high_low_ratio'] = (high - low) / close
        features['close_open_ratio'] = (close - open_price) / open_price

        # 13. 시간 특징
        features['hour'] = df.index.hour
        features['day_of_week'] = df.index.dayofweek

        # NaN 처리
        features = features.fillna(method='ffill').fillna(0)

        return features

    def create_labels_with_noise_filter(self, df, timeframe):
        """노이즈 필터링된 라벨 생성"""
        # 타임프레임별 적응적 임계값
        thresholds = {
            '15m': 0.0015,  # 0.15%
            '30m': 0.002,   # 0.2%
            '1h': 0.003,    # 0.3%
            '4h': 0.005     # 0.5%
        }

        threshold = thresholds.get(timeframe, 0.002)

        # 미래 수익률 (여러 기간 고려)
        future_returns = pd.DataFrame(index=df.index)
        future_returns['r1'] = df['close'].shift(-1) / df['close'] - 1
        future_returns['r2'] = df['close'].shift(-2) / df['close'] - 1
        future_returns['r3'] = df['close'].shift(-3) / df['close'] - 1

        # 가중 평균 미래 수익률
        weighted_return = (future_returns['r1'] * 0.5 +
                          future_returns['r2'] * 0.3 +
                          future_returns['r3'] * 0.2)

        # 라벨 생성 (명확한 신호만)
        labels = pd.Series(index=df.index, dtype=int)
        labels[weighted_return > threshold] = 1  # UP
        labels[weighted_return < -threshold] = 0  # DOWN

        # 노이즈 제거 (임계값 내 변동은 제외)
        labels[(weighted_return >= -threshold) & (weighted_return <= threshold)] = np.nan

        return labels

    def train_ensemble_model(self, timeframe):
        """앙상블 모델 훈련"""
        print(f"\n{'='*60}")
        print(f"🚀 {timeframe} 앙상블 모델 훈련")
        print(f"{'='*60}")

        # 데이터 수집
        df = self.get_data(timeframe, limit=10000)

        # 특징 생성
        features = self.create_advanced_features(df)

        # 라벨 생성
        labels = self.create_labels_with_noise_filter(df, timeframe)

        # 유효한 데이터만 사용
        valid_idx = ~(features.isna().any(axis=1) | labels.isna())
        X = features[valid_idx]
        y = labels[valid_idx]

        print(f"훈련 데이터: {len(X)}개 샘플")
        print(f"UP: {(y==1).sum()}개, DOWN: {(y==0).sum()}개")

        # 시계열 분할
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        # 스케일링
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # 개별 모델 정의
        rf_model = RandomForestClassifier(
            n_estimators=300,
            max_depth=15,
            min_samples_split=10,
            min_samples_leaf=5,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1
        )

        gb_model = GradientBoostingClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=7,
            min_samples_split=10,
            min_samples_leaf=5,
            subsample=0.8,
            random_state=42
        )

        xgb_model = XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=7,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss'
        )

        nn_model = MLPClassifier(
            hidden_layer_sizes=(100, 50, 25),
            activation='relu',
            learning_rate='adaptive',
            learning_rate_init=0.001,
            max_iter=1000,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=42
        )

        # 앙상블 모델 (Voting)
        ensemble = VotingClassifier(
            estimators=[
                ('rf', rf_model),
                ('gb', gb_model),
                ('xgb', xgb_model),
                ('nn', nn_model)
            ],
            voting='soft',  # 확률 기반 투표
            weights=[1, 1.5, 1.5, 1]  # XGBoost와 GradientBoosting에 더 높은 가중치
        )

        # 훈련
        print("앙상블 모델 훈련 중...")
        ensemble.fit(X_train_scaled, y_train)

        # 평가
        y_pred_train = ensemble.predict(X_train_scaled)
        y_pred_test = ensemble.predict(X_test_scaled)
        y_proba_test = ensemble.predict_proba(X_test_scaled)

        # 메트릭 계산
        train_acc = accuracy_score(y_train, y_pred_train)
        test_acc = accuracy_score(y_test, y_pred_test)
        precision = precision_score(y_test, y_pred_test)
        recall = recall_score(y_test, y_pred_test)
        f1 = f1_score(y_test, y_pred_test)

        print(f"\n📊 모델 평가:")
        print(f"훈련 정확도: {train_acc*100:.1f}%")
        print(f"테스트 정확도: {test_acc*100:.1f}%")
        print(f"정밀도: {precision*100:.1f}%")
        print(f"재현율: {recall*100:.1f}%")
        print(f"F1 점수: {f1:.3f}")

        # 개별 모델 성능
        print(f"\n📈 개별 모델 성능:")
        for estimator in ensemble.estimators_:
            name = estimator[0]
            model = estimator[1]
            model.fit(X_train_scaled, y_train)
            individual_acc = accuracy_score(y_test, model.predict(X_test_scaled))
            print(f"  {name.upper()}: {individual_acc*100:.1f}%")

        # 예측 분포
        pred_dist = pd.Series(y_pred_test).value_counts()
        print(f"\n예측 분포:")
        print(f"  DOWN: {pred_dist.get(0, 0)}개 ({pred_dist.get(0, 0)/len(y_pred_test)*100:.1f}%)")
        print(f"  UP: {pred_dist.get(1, 0)}개 ({pred_dist.get(1, 0)/len(y_pred_test)*100:.1f}%)")

        # 신뢰도 분석
        confidence_scores = np.max(y_proba_test, axis=1)
        print(f"\n신뢰도 분석:")
        print(f"  평균: {confidence_scores.mean():.1%}")
        print(f"  최소: {confidence_scores.min():.1%}")
        print(f"  최대: {confidence_scores.max():.1%}")

        # 모델 저장
        if test_acc > 0.55:  # 55% 이상인 경우만 저장
            model_info = {
                'model': ensemble,
                'scaler': scaler,
                'features': list(features.columns),
                'accuracy': test_acc,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'timeframe': timeframe,
                'trained_at': datetime.now().isoformat()
            }

            filename = f"improved_{timeframe}_ensemble_model.pkl"
            joblib.dump(model_info, f"models/{filename}")
            print(f"\n✅ 모델 저장: models/{filename}")

            return model_info
        else:
            print(f"\n⚠️ 정확도가 낮아 저장하지 않음 ({test_acc*100:.1f}%)")
            return None

def main():
    trainer = ImprovedModelTrainer()

    print("=" * 60)
    print("🔧 개선된 모델 훈련 시작")
    print("=" * 60)

    results = {}
    for timeframe in ['15m', '30m', '1h', '4h']:
        model_info = trainer.train_ensemble_model(timeframe)
        if model_info:
            results[timeframe] = model_info

    # 결과 요약
    print("\n" + "=" * 60)
    print("📋 훈련 결과 요약")
    print("=" * 60)

    for tf, info in results.items():
        print(f"\n{tf}:")
        print(f"  정확도: {info['accuracy']*100:.1f}%")
        print(f"  F1 점수: {info['f1_score']:.3f}")

if __name__ == "__main__":
    main()