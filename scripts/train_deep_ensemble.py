#!/usr/bin/env python3
"""
Deep Ensemble Models with Time-Specific Features
시간대별 상승/하락 특화 - 충분한 데이터로 60% 이상 목표
"""

import ccxt
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import joblib
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class DeepEnsembleTrainer:
    def __init__(self):
        self.exchange = ccxt.binance()

    def get_massive_data(self, timeframe, days=45):
        """대량 데이터 수집 (45일)"""
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

        collected = 0
        while collected < total_candles:
            try:
                since = end_time - (collected + chunk_size) * ms_per_candle
                ohlcv = self.exchange.fetch_ohlcv('BTC/USDT', timeframe, since=since, limit=chunk_size)

                if not ohlcv:
                    break

                all_data = ohlcv + all_data
                collected = len(all_data)

                if collected % 2000 == 0:
                    print(f"  수집 진행: {collected}/{total_candles} ({collected/total_candles*100:.1f}%)")

                if collected >= total_candles:
                    all_data = all_data[-total_candles:]
                    break

            except Exception as e:
                print(f"  ⚠️ 수집 오류, 재시도...")
                continue

        df = pd.DataFrame(all_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)

        print(f"  ✅ {len(df)}개 캔들 수집 완료")
        return df

    def create_time_specific_features(self, df, timeframe, direction='up'):
        """시간대별 특화 특징"""
        features = pd.DataFrame(index=df.index)

        # 1. 기본 수익률 (다양한 기간)
        periods = [1, 2, 3, 5, 8, 13, 21, 34, 55]
        for period in periods:
            features[f'return_{period}'] = df['close'].pct_change(period)
            features[f'log_return_{period}'] = np.log(df['close'] / df['close'].shift(period))

        # 2. RSI 변형
        for period in [7, 14, 21, 28]:
            delta = df['close'].diff()
            gain = delta.where(delta > 0, 0).rolling(period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
            rsi = 100 - (100 / (1 + gain / (loss + 1e-10)))
            features[f'rsi_{period}'] = rsi
            features[f'rsi_{period}_sma'] = rsi.rolling(5).mean()
            features[f'rsi_{period}_std'] = rsi.rolling(10).std()

        # 3. 이동평균 조합
        sma_periods = [5, 10, 20, 50, 100, 200]
        for i, period1 in enumerate(sma_periods[:-1]):
            for period2 in sma_periods[i+1:]:
                sma1 = df['close'].rolling(window=period1).mean()
                sma2 = df['close'].rolling(window=period2).mean()
                features[f'sma_cross_{period1}_{period2}'] = (sma1 - sma2) / df['close']

        # 4. 볼린저 밴드 다양화
        for period in [10, 20, 30]:
            for std_mult in [1.5, 2, 2.5]:
                ma = df['close'].rolling(window=period).mean()
                std = df['close'].rolling(window=period).std()
                features[f'bb_pos_{period}_{std_mult}'] = (df['close'] - ma) / (std_mult * std + 1e-10)

        # 5. MACD 변형
        for fast, slow, signal in [(12, 26, 9), (5, 35, 5), (10, 20, 5)]:
            exp_fast = df['close'].ewm(span=fast, adjust=False).mean()
            exp_slow = df['close'].ewm(span=slow, adjust=False).mean()
            macd = exp_fast - exp_slow
            macd_signal = macd.ewm(span=signal, adjust=False).mean()
            features[f'macd_{fast}_{slow}'] = macd / df['close']
            features[f'macd_hist_{fast}_{slow}'] = (macd - macd_signal) / df['close']

        # 6. 볼륨 지표
        features['volume_sma_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
        features['volume_ema_ratio'] = df['volume'] / df['volume'].ewm(span=20).mean()
        features['volume_rsi'] = self.calculate_rsi(df['volume'], 14)
        features['price_volume_trend'] = (df['close'].pct_change() * df['volume']).rolling(20).sum()

        # 7. 변동성 지표
        for period in [10, 20, 30]:
            returns = df['close'].pct_change()
            features[f'volatility_{period}'] = returns.rolling(window=period).std()
            features[f'volatility_ratio_{period}'] = returns.rolling(window=period).std() / returns.rolling(window=period*2).std()

        # 8. 가격 패턴
        for period in [5, 10, 20]:
            features[f'high_low_ratio_{period}'] = df['high'].rolling(period).max() / df['low'].rolling(period).min()
            features[f'close_to_high_{period}'] = df['close'] / df['high'].rolling(period).max()
            features[f'close_to_low_{period}'] = df['close'] / df['low'].rolling(period).min()

        # 9. 시간대별 특화 (timeframe specific)
        if timeframe == '15m':
            # 15분봉: 단기 모멘텀 중심
            features['micro_momentum'] = df['close'].pct_change(4).rolling(8).mean()
            features['quick_reversal'] = (df['close'].pct_change() * df['close'].pct_change().shift()).rolling(4).sum()
            features['volume_burst'] = (df['volume'] > df['volume'].rolling(20).mean() * 1.5).rolling(4).sum()

        elif timeframe == '30m':
            # 30분봉: 중단기 트렌드
            features['trend_strength'] = self.calculate_trend_strength(df, 20)
            features['trend_consistency'] = (df['close'].pct_change() > 0).rolling(10).mean()
            features['breakout_potential'] = (df['high'] - df['high'].rolling(20).mean()) / df['high'].rolling(20).std()

        elif timeframe == '1h':
            # 1시간봉: 일중 패턴
            features['intraday_range'] = (df['high'].rolling(24).max() - df['low'].rolling(24).min()) / df['close']
            features['session_momentum'] = df['close'].pct_change(8).rolling(3).mean()
            features['hourly_vwap'] = (df['close'] * df['volume']).rolling(24).sum() / df['volume'].rolling(24).sum()

        elif timeframe == '4h':
            # 4시간봉: 장기 트렌드
            features['major_trend'] = (df['close'].rolling(50).mean() - df['close'].rolling(200).mean()) / df['close']
            features['trend_acceleration'] = df['close'].rolling(20).mean().pct_change(10)
            features['long_term_support'] = df['close'] / df['low'].rolling(100).min()

        # 10. 방향별 특화 (direction specific)
        if direction == 'up':
            # 상승 특화 지표
            features['bullish_pressure'] = (df['close'] - df['open']).where(df['close'] > df['open'], 0).rolling(10).sum() / df['close']
            features['higher_highs'] = (df['high'] > df['high'].rolling(10).max().shift()).rolling(10).sum()
            features['dip_buying'] = ((df['low'] <= df['low'].rolling(20).min()) & (df['close'] > df['open'])).rolling(20).sum()
            features['accumulation'] = self.calculate_accumulation(df)

        else:
            # 하락 특화 지표
            features['bearish_pressure'] = (df['open'] - df['close']).where(df['close'] < df['open'], 0).rolling(10).sum() / df['close']
            features['lower_lows'] = (df['low'] < df['low'].rolling(10).min().shift()).rolling(10).sum()
            features['resistance_failure'] = ((df['high'] >= df['high'].rolling(20).max()) & (df['close'] < df['open'])).rolling(20).sum()
            features['distribution'] = self.calculate_distribution(df)

        # 11. 시간 특징
        features['hour'] = df.index.hour
        features['day_of_week'] = df.index.dayofweek
        features['is_weekend'] = (df.index.dayofweek >= 5).astype(int)

        # 시간대별 활동성
        features['asia_session'] = ((df.index.hour >= 0) & (df.index.hour < 8)).astype(int)
        features['europe_session'] = ((df.index.hour >= 8) & (df.index.hour < 16)).astype(int)
        features['us_session'] = ((df.index.hour >= 16) & (df.index.hour < 24)).astype(int)

        # Clean data
        features = features.replace([np.inf, -np.inf], np.nan)
        features = features.fillna(0)

        return features

    def calculate_rsi(self, series, period):
        """RSI 계산"""
        delta = series.diff()
        gain = delta.where(delta > 0, 0).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        return 100 - (100 / (1 + gain / (loss + 1e-10)))

    def calculate_trend_strength(self, df, period):
        """트렌드 강도 계산"""
        returns = df['close'].pct_change()
        positive_returns = returns.where(returns > 0, 0).rolling(period).sum()
        negative_returns = returns.where(returns < 0, 0).abs().rolling(period).sum()
        return (positive_returns - negative_returns) / (positive_returns + negative_returns + 1e-10)

    def calculate_accumulation(self, df):
        """축적 지표"""
        volume_price = ((df['close'] - df['low']) / (df['high'] - df['low'] + 1e-10) * df['volume'])
        return volume_price.rolling(20).sum() / df['volume'].rolling(20).sum()

    def calculate_distribution(self, df):
        """분산 지표"""
        volume_price = ((df['high'] - df['close']) / (df['high'] - df['low'] + 1e-10) * df['volume'])
        return volume_price.rolling(20).sum() / df['volume'].rolling(20).sum()

    def create_labels(self, df, direction='up', timeframe='15m'):
        """방향별 라벨 생성"""
        # 시간대별 예측 horizon
        horizons = {
            '15m': 8,   # 2시간
            '30m': 4,   # 2시간
            '1h': 2,    # 2시간
            '4h': 1,    # 4시간
        }

        horizon = horizons.get(timeframe, 4)
        future_return = df['close'].shift(-horizon) / df['close'] - 1

        # 시간대별 임계값
        thresholds = {
            '15m': 0.002,  # 0.2%
            '30m': 0.003,  # 0.3%
            '1h': 0.004,   # 0.4%
            '4h': 0.005,   # 0.5%
        }

        threshold = thresholds.get(timeframe, 0.003)

        if direction == 'up':
            labels = (future_return > threshold).astype(int)
        else:
            labels = (future_return < -threshold).astype(int)

        return labels

    def train_deep_ensemble(self, timeframe, direction='up'):
        """Deep Ensemble 모델 훈련"""
        print(f"\n{'='*60}")
        print(f"🚀 {timeframe} {direction.upper()} Deep Ensemble 훈련")
        print(f"{'='*60}")

        # 1. 대량 데이터 수집
        df = self.get_massive_data(timeframe, days=45)

        # 2. 특징 생성
        print(f"  📐 시간대별 특화 특징 생성...")
        features = self.create_time_specific_features(df, timeframe, direction)
        labels = self.create_labels(df, direction, timeframe)

        # 3. 데이터 정리
        valid_idx = ~(features.isna().any(axis=1) | labels.isna())
        X = features[valid_idx]
        y = labels[valid_idx]

        print(f"  📊 데이터: {len(X)}개 샘플, {X.shape[1]}개 특징")
        print(f"  📈 타겟 비율: {y.mean():.1%}")

        # 4. Train/Test Split
        split_idx = int(len(X) * 0.8)
        X_train = X[:split_idx]
        X_test = X[split_idx:]
        y_train = y[:split_idx]
        y_test = y[split_idx:]

        # 5. 스케일링
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # 6. 모델 정의 (깊은 앙상블)
        models = []

        # XGBoost variants
        for depth in [3, 5, 7]:
            for n_est in [300, 500]:
                xgb = XGBClassifier(
                    n_estimators=n_est,
                    max_depth=depth,
                    learning_rate=0.03,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    gamma=0.1,
                    reg_alpha=0.1,
                    reg_lambda=1,
                    random_state=42 + depth + n_est,
                    use_label_encoder=False,
                    eval_metric='logloss',
                    n_jobs=-1
                )
                models.append((f'xgb_{depth}_{n_est}', xgb))

        # LightGBM variants
        for leaves in [31, 63, 127]:
            for n_est in [300, 500]:
                lgb = LGBMClassifier(
                    n_estimators=n_est,
                    num_leaves=leaves,
                    max_depth=-1,
                    learning_rate=0.03,
                    min_child_samples=20,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    reg_alpha=0.1,
                    reg_lambda=1,
                    random_state=42 + leaves + n_est,
                    verbosity=-1,
                    n_jobs=-1
                )
                models.append((f'lgb_{leaves}_{n_est}', lgb))

        # CatBoost
        cat = CatBoostClassifier(
            iterations=500,
            depth=6,
            learning_rate=0.03,
            l2_leaf_reg=3,
            random_seed=42,
            verbose=False
        )
        models.append(('catboost', cat))

        # Extra Trees
        et = ExtraTreesClassifier(
            n_estimators=500,
            max_depth=20,
            min_samples_split=20,
            min_samples_leaf=10,
            random_state=42,
            n_jobs=-1
        )
        models.append(('extra_trees', et))

        print(f"  🎯 {len(models)}개 모델로 앙상블 구성")

        # 7. 각 모델 훈련 및 평가
        model_results = {}
        all_predictions = []
        all_probabilities = []

        for name, model in models:
            print(f"  🔧 {name} 훈련 중...")

            # Train
            model.fit(X_train_scaled, y_train)

            # Predict
            y_pred = model.predict(X_test_scaled)
            y_proba = model.predict_proba(X_test_scaled)[:, 1]

            # Metrics
            acc = accuracy_score(y_test, y_pred)
            model_results[name] = acc

            all_predictions.append(y_pred)
            all_probabilities.append(y_proba)

            if acc >= 0.55:
                print(f"    ✅ {name}: {acc:.1%}")
            else:
                print(f"    ⚠️ {name}: {acc:.1%}")

        # 8. 앙상블 예측
        print(f"\n  🎯 앙상블 예측...")

        # Weighted voting based on individual performance
        weights = np.array([model_results[name] for name, _ in models])
        weights = weights / weights.sum()

        ensemble_proba = np.average(all_probabilities, axis=0, weights=weights)
        ensemble_pred = (ensemble_proba > 0.5).astype(int)

        # Final metrics
        ensemble_acc = accuracy_score(y_test, ensemble_pred)
        ensemble_prec = precision_score(y_test, ensemble_pred, zero_division=0)
        ensemble_rec = recall_score(y_test, ensemble_pred, zero_division=0)
        ensemble_f1 = f1_score(y_test, ensemble_pred, zero_division=0)
        ensemble_auc = roc_auc_score(y_test, ensemble_proba)

        print(f"  📊 앙상블 결과:")
        print(f"     정확도: {ensemble_acc:.1%}")
        print(f"     정밀도: {ensemble_prec:.1%}")
        print(f"     재현율: {ensemble_rec:.1%}")
        print(f"     F1 점수: {ensemble_f1:.3f}")
        print(f"     AUC: {ensemble_auc:.3f}")

        # 9. 높은 신뢰도 예측 분석
        high_conf_threshold = 0.65
        high_conf_idx = np.where((ensemble_proba > high_conf_threshold) | (ensemble_proba < (1-high_conf_threshold)))[0]
        if len(high_conf_idx) > 0:
            high_conf_acc = accuracy_score(y_test.iloc[high_conf_idx], ensemble_pred[high_conf_idx])
            print(f"  💎 높은 신뢰도 (>{high_conf_threshold:.0%}) 정확도: {high_conf_acc:.1%} ({len(high_conf_idx)}개)")

        # 10. 모델 저장
        model_info = {
            'models': [(name, model) for name, model in models],
            'weights': weights,
            'scaler': scaler,
            'features': list(features.columns),
            'timeframe': timeframe,
            'direction': direction,
            'ensemble_accuracy': ensemble_acc,
            'ensemble_auc': ensemble_auc,
            'model_results': model_results,
            'trained_at': datetime.now().isoformat()
        }

        filename = f"deep_ensemble_{timeframe}_{direction}_model.pkl"
        joblib.dump(model_info, f"models/{filename}")
        print(f"  ✅ 모델 저장: models/{filename}")

        return {
            'timeframe': timeframe,
            'direction': direction,
            'accuracy': ensemble_acc,
            'auc': ensemble_auc,
            'f1': ensemble_f1
        }

def main():
    print("="*60)
    print("🚀 Deep Ensemble 모델 훈련")
    print("📅 " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("💪 시간대별 상승/하락 특화 (충분한 데이터)")
    print("="*60)

    trainer = DeepEnsembleTrainer()
    results = []

    # 각 시간대별 상승/하락 모델 훈련
    for timeframe in ['15m', '30m', '1h', '4h']:
        for direction in ['up', 'down']:
            try:
                result = trainer.train_deep_ensemble(timeframe, direction)
                results.append(result)
            except Exception as e:
                print(f"❌ {timeframe} {direction} 훈련 실패: {e}")

    # 결과 요약
    print("\n" + "="*60)
    print("📊 Deep Ensemble 훈련 결과")
    print("="*60)

    for result in results:
        emoji = "✅" if result['accuracy'] >= 0.6 else "⚠️" if result['accuracy'] >= 0.55 else "❌"
        print(f"{emoji} {result['timeframe']} {result['direction'].upper()}: "
              f"Acc={result['accuracy']:.1%}, AUC={result['auc']:.3f}, F1={result['f1']:.3f}")

    # 평가
    avg_accuracy = np.mean([r['accuracy'] for r in results])
    print(f"\n평균 정확도: {avg_accuracy:.1%}")

    if avg_accuracy >= 0.6:
        print("🎉 목표 달성! 60% 이상 정확도")
    elif avg_accuracy >= 0.55:
        print("📈 개선 중... 추가 튜닝 필요")
    else:
        print("🔧 더 많은 개선 필요")

if __name__ == "__main__":
    main()