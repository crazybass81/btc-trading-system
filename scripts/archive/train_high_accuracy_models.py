#!/usr/bin/env python3
"""
고정확도 ML 모델 훈련 스크립트
- 대량 데이터 (30K+ 캔들)
- 고급 특징 공학
- 하이퍼파라미터 최적화
- 교차 검증
- 앙상블 방법
"""

import ccxt
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import joblib
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class HighAccuracyTrainer:
    def __init__(self):
        self.exchange = ccxt.binance()
        self.best_models = {}

    def get_extended_data(self, timeframe, limit=30000):
        """대량 데이터 수집 (Binance API 제한으로 나누어 수집)"""
        print(f"📊 대량 데이터 수집 시작: {timeframe}")

        all_data = []
        max_limit = 1000  # Binance API 최대 한계

        # 여러 번 호출하여 데이터 수집
        for i in range(limit // max_limit):
            try:
                since = None
                if all_data:
                    # 마지막 타임스탬프부터 이어서 수집
                    since = all_data[-1][0] - (max_limit * 60000)  # 밀리초 단위

                ohlcv = self.exchange.fetch_ohlcv(
                    'BTC/USDT',
                    timeframe=timeframe,
                    limit=max_limit,
                    since=since
                )

                # 중복 제거하며 추가
                if all_data:
                    # 중복되는 마지막 항목 제거
                    ohlcv = [x for x in ohlcv if x[0] < all_data[0][0]]

                all_data = ohlcv + all_data

                print(f"  수집 진행: {len(all_data)}/{limit} ({len(all_data)/limit*100:.1f}%)")

                if len(all_data) >= limit:
                    break

            except Exception as e:
                print(f"  ⚠️ 데이터 수집 오류: {e}")
                break

        # DataFrame 변환
        df = pd.DataFrame(all_data[:limit], columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)

        # 기간 정보
        if len(df) > 0:
            start_date = df.index[0].strftime('%Y-%m-%d')
            end_date = df.index[-1].strftime('%Y-%m-%d')
            print(f"  ✅ 수집 완료: {len(df)}개 캔들 ({start_date} ~ {end_date})")

        return df

    def create_advanced_features(self, df):
        """고급 특징 공학 (100+ 특징)"""
        features = pd.DataFrame(index=df.index)

        # 1. 가격 변화율 (다양한 기간)
        for period in [1, 2, 3, 5, 8, 13, 21, 34, 55, 89]:  # 피보나치 수열
            features[f'return_{period}'] = df['close'].pct_change(period)
            features[f'return_{period}_abs'] = abs(features[f'return_{period}'])

        # 2. 이동평균 (단순, 지수)
        for period in [5, 10, 20, 50, 100, 200]:
            sma = df['close'].rolling(window=period).mean()
            ema = df['close'].ewm(span=period, adjust=False).mean()

            features[f'sma_{period}'] = (df['close'] - sma) / sma
            features[f'ema_{period}'] = (df['close'] - ema) / ema
            features[f'sma_ema_diff_{period}'] = (sma - ema) / df['close']

            # 이동평균 기울기
            features[f'sma_{period}_slope'] = sma.pct_change(5)
            features[f'ema_{period}_slope'] = ema.pct_change(5)

        # 3. RSI 변형
        for period in [7, 14, 21, 28]:
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / (loss + 1e-10)
            rsi = 100 - (100 / (1 + rs))

            features[f'rsi_{period}'] = rsi
            features[f'rsi_{period}_ma'] = rsi.rolling(window=5).mean()
            features[f'rsi_{period}_std'] = rsi.rolling(window=5).std()

        # 4. MACD 변형
        for fast, slow in [(12, 26), (5, 35), (10, 30)]:
            exp1 = df['close'].ewm(span=fast, adjust=False).mean()
            exp2 = df['close'].ewm(span=slow, adjust=False).mean()
            macd = exp1 - exp2
            signal = macd.ewm(span=9, adjust=False).mean()

            features[f'macd_{fast}_{slow}'] = macd / df['close']
            features[f'macd_signal_{fast}_{slow}'] = signal / df['close']
            features[f'macd_hist_{fast}_{slow}'] = (macd - signal) / df['close']

        # 5. 볼린저 밴드 (다양한 설정)
        for period in [10, 20, 50]:
            for num_std in [1.5, 2, 2.5, 3]:
                ma = df['close'].rolling(window=period).mean()
                std = df['close'].rolling(window=period).std()

                upper = ma + (num_std * std)
                lower = ma - (num_std * std)

                features[f'bb_{period}_{num_std}_pos'] = (df['close'] - lower) / (upper - lower)
                features[f'bb_{period}_{num_std}_width'] = (upper - lower) / ma

        # 6. 스토캐스틱
        for period in [5, 14, 21]:
            for smooth in [3, 5]:
                low_min = df['low'].rolling(window=period).min()
                high_max = df['high'].rolling(window=period).max()

                k = 100 * ((df['close'] - low_min) / (high_max - low_min + 1e-10))
                d = k.rolling(window=smooth).mean()

                features[f'stoch_k_{period}_{smooth}'] = k
                features[f'stoch_d_{period}_{smooth}'] = d
                features[f'stoch_diff_{period}_{smooth}'] = k - d

        # 7. ATR과 변동성
        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift())
        low_close = abs(df['low'] - df['close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)

        for period in [7, 14, 21, 28]:
            atr = tr.rolling(window=period).mean()
            features[f'atr_{period}'] = atr / df['close']
            features[f'atr_{period}_ma'] = features[f'atr_{period}'].rolling(window=5).mean()

        # 8. 볼륨 지표
        features['volume_ratio'] = df['volume'] / df['volume'].rolling(window=20).mean()
        features['volume_trend'] = df['volume'].rolling(window=10).mean().pct_change(5)

        # OBV (On Balance Volume)
        obv = (df['volume'] * ((df['close'] - df['close'].shift()) > 0).astype(int) -
               df['volume'] * ((df['close'] - df['close'].shift()) < 0).astype(int)).cumsum()
        features['obv'] = obv / obv.rolling(window=20).mean()
        features['obv_slope'] = obv.pct_change(5)

        # VWAP (Volume Weighted Average Price)
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        vwap = (typical_price * df['volume']).rolling(window=20).sum() / df['volume'].rolling(window=20).sum()
        features['vwap_ratio'] = df['close'] / vwap

        # 9. 패턴 인식
        # 캔들 패턴
        body = df['close'] - df['open']
        upper_shadow = df['high'] - df[['close', 'open']].max(axis=1)
        lower_shadow = df[['close', 'open']].min(axis=1) - df['low']

        features['body_ratio'] = body / (df['high'] - df['low'] + 1e-10)
        features['upper_shadow_ratio'] = upper_shadow / (df['high'] - df['low'] + 1e-10)
        features['lower_shadow_ratio'] = lower_shadow / (df['high'] - df['low'] + 1e-10)

        # Doji 패턴
        features['doji'] = (abs(body) / (df['high'] - df['low'] + 1e-10) < 0.1).astype(int)

        # Hammer 패턴
        features['hammer'] = ((lower_shadow > 2 * abs(body)) &
                             (upper_shadow < abs(body) * 0.3)).astype(int)

        # 10. 지지/저항선
        for period in [20, 50, 100]:
            features[f'dist_from_high_{period}'] = (df['high'].rolling(window=period).max() - df['close']) / df['close']
            features[f'dist_from_low_{period}'] = (df['close'] - df['low'].rolling(window=period).min()) / df['close']

        # 11. 시간 특징
        features['hour'] = df.index.hour
        features['day_of_week'] = df.index.dayofweek
        features['day_of_month'] = df.index.day
        features['month'] = df.index.month

        # 12. 시장 마이크로구조
        # 틱 규칙
        features['tick_rule'] = np.sign(df['close'].diff())

        # 스프레드 프록시
        features['spread_proxy'] = (df['high'] - df['low']) / df['close']

        # 13. 롤링 통계
        for period in [10, 20, 50]:
            returns = df['close'].pct_change()

            features[f'return_mean_{period}'] = returns.rolling(window=period).mean()
            features[f'return_std_{period}'] = returns.rolling(window=period).std()
            features[f'return_skew_{period}'] = returns.rolling(window=period).skew()
            features[f'return_kurt_{period}'] = returns.rolling(window=period).kurt()

        return features.fillna(0)

    def create_optimized_labels(self, df, timeframe):
        """최적화된 라벨 생성"""
        # 타임프레임별 적응적 임계값
        thresholds = {
            '15m': 0.0015,  # 0.15%
            '30m': 0.002,   # 0.2%
            '1h': 0.0025,   # 0.25%
            '4h': 0.004     # 0.4%
        }

        threshold = thresholds.get(timeframe, 0.002)

        # 다중 기간 미래 수익률 고려
        future_returns = pd.DataFrame(index=df.index)

        # 가중 평균 방식
        weights = [0.5, 0.3, 0.15, 0.05]  # 가까운 미래에 더 높은 가중치

        for i, weight in enumerate(weights, 1):
            future_returns[f'r{i}'] = df['close'].shift(-i) / df['close'] - 1

        weighted_return = sum(future_returns[f'r{i+1}'] * weight
                             for i, weight in enumerate(weights))

        # 명확한 신호만 라벨링
        labels = pd.Series(index=df.index, dtype=int)
        labels[weighted_return > threshold] = 1  # UP
        labels[weighted_return < -threshold] = 0  # DOWN
        labels[(weighted_return >= -threshold) & (weighted_return <= threshold)] = -1  # 제외

        # -1 제거 (노이즈)
        labels = labels[labels != -1]

        return labels

    def optimize_hyperparameters(self, X_train, y_train, model_type='xgboost'):
        """GridSearch를 통한 하이퍼파라미터 최적화"""
        print(f"  🔧 {model_type} 하이퍼파라미터 최적화 중...")

        # TimeSeriesSplit for cross-validation
        tscv = TimeSeriesSplit(n_splits=5)

        if model_type == 'xgboost':
            model = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
            param_grid = {
                'n_estimators': [300, 500],
                'max_depth': [5, 7, 10],
                'learning_rate': [0.01, 0.05, 0.1],
                'subsample': [0.7, 0.8, 0.9],
                'colsample_bytree': [0.7, 0.8, 0.9]
            }

        elif model_type == 'lightgbm':
            model = LGBMClassifier(random_state=42, verbosity=-1)
            param_grid = {
                'n_estimators': [300, 500],
                'max_depth': [5, 7, 10],
                'learning_rate': [0.01, 0.05, 0.1],
                'num_leaves': [31, 50, 100],
                'min_child_samples': [20, 30, 40]
            }

        elif model_type == 'catboost':
            model = CatBoostClassifier(random_state=42, verbose=False)
            param_grid = {
                'iterations': [300, 500],
                'depth': [4, 6, 8],
                'learning_rate': [0.01, 0.05, 0.1],
                'l2_leaf_reg': [1, 3, 5]
            }

        else:  # random forest
            model = RandomForestClassifier(random_state=42, n_jobs=-1)
            param_grid = {
                'n_estimators': [300, 500],
                'max_depth': [10, 15, 20],
                'min_samples_split': [5, 10, 20],
                'min_samples_leaf': [2, 5, 10],
                'max_features': ['sqrt', 'log2', 0.3]
            }

        # GridSearch
        grid_search = GridSearchCV(
            model,
            param_grid,
            cv=tscv,
            scoring='accuracy',
            n_jobs=-1,
            verbose=0
        )

        grid_search.fit(X_train, y_train)

        print(f"    최적 파라미터: {grid_search.best_params_}")
        print(f"    최고 CV 점수: {grid_search.best_score_:.3f}")

        return grid_search.best_estimator_

    def feature_selection(self, X, y, n_features=50):
        """특징 선택"""
        print(f"  🔍 특징 선택 중 (상위 {n_features}개)...")

        # 1. Univariate Selection
        selector = SelectKBest(f_classif, k=n_features)
        selector.fit(X, y)

        # 특징 점수
        scores = pd.DataFrame({
            'feature': X.columns,
            'score': selector.scores_
        }).sort_values('score', ascending=False)

        selected_features = scores.head(n_features)['feature'].tolist()

        print(f"    선택된 주요 특징: {selected_features[:5]}")

        return selected_features

    def train_ensemble_model(self, timeframe, data_limit=30000):
        """앙상블 모델 훈련"""
        print(f"\n{'='*60}")
        print(f"🚀 {timeframe} 고정확도 앙상블 모델 훈련")
        print(f"{'='*60}")

        # 1. 대량 데이터 수집
        df = self.get_extended_data(timeframe, limit=data_limit)

        if len(df) < 1000:
            print(f"  ⚠️ 데이터 부족: {len(df)}개")
            return None

        # 2. 고급 특징 생성
        print("  📐 고급 특징 생성 중...")
        features = self.create_advanced_features(df)

        # 3. 라벨 생성
        labels = self.create_optimized_labels(df, timeframe)

        # 유효 데이터
        valid_idx = features.index.intersection(labels.index)
        X = features.loc[valid_idx]
        y = labels.loc[valid_idx]

        # NaN 제거
        valid_mask = ~(X.isna().any(axis=1) | y.isna())
        X = X[valid_mask]
        y = y[valid_mask]

        print(f"  📊 데이터: {len(X)}개 샘플, {len(features.columns)}개 특징")
        print(f"  📈 UP: {(y==1).sum()}개, DOWN: {(y==0).sum()}개")

        # 4. 특징 선택
        selected_features = self.feature_selection(X, y, n_features=50)
        X_selected = X[selected_features]

        # 5. 데이터 분할
        split_idx = int(len(X_selected) * 0.8)
        X_train, X_test = X_selected.iloc[:split_idx], X_selected.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        # 6. 스케일링
        scaler = RobustScaler()  # 이상치에 강한 스케일러
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # 7. 여러 모델 최적화 및 훈련
        models = {}

        # XGBoost
        models['xgboost'] = self.optimize_hyperparameters(X_train_scaled, y_train, 'xgboost')

        # LightGBM
        models['lightgbm'] = self.optimize_hyperparameters(X_train_scaled, y_train, 'lightgbm')

        # CatBoost
        models['catboost'] = self.optimize_hyperparameters(X_train_scaled, y_train, 'catboost')

        # Random Forest
        models['rf'] = self.optimize_hyperparameters(X_train_scaled, y_train, 'rf')

        # 8. 개별 모델 평가
        print("\n  📊 개별 모델 성능:")
        best_model = None
        best_score = 0

        for name, model in models.items():
            y_pred = model.predict(X_test_scaled)
            y_proba = model.predict_proba(X_test_scaled)[:, 1]

            acc = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)

            try:
                auc = roc_auc_score(y_test, y_proba)
            except:
                auc = 0.5

            print(f"    {name:10s}: Acc={acc:.3f}, P={precision:.3f}, R={recall:.3f}, F1={f1:.3f}, AUC={auc:.3f}")

            # 최고 모델 선택
            score = acc * 0.3 + precision * 0.2 + recall * 0.2 + f1 * 0.2 + auc * 0.1
            if score > best_score:
                best_score = score
                best_model = (name, model)

        # 9. 앙상블 예측 (Voting)
        print("\n  🎯 앙상블 예측 (소프트 보팅):")

        # 각 모델의 확률 예측
        predictions = np.zeros((len(X_test_scaled), 2))

        weights = {'xgboost': 1.5, 'lightgbm': 1.5, 'catboost': 1.2, 'rf': 1.0}
        total_weight = sum(weights.values())

        for name, model in models.items():
            proba = model.predict_proba(X_test_scaled)
            predictions += proba * weights[name] / total_weight

        # 앙상블 최종 예측
        y_ensemble = (predictions[:, 1] > 0.5).astype(int)

        ensemble_acc = accuracy_score(y_test, y_ensemble)
        ensemble_precision = precision_score(y_test, y_ensemble, zero_division=0)
        ensemble_recall = recall_score(y_test, y_ensemble, zero_division=0)
        ensemble_f1 = f1_score(y_test, y_ensemble, zero_division=0)

        print(f"    앙상블: Acc={ensemble_acc:.3f}, P={ensemble_precision:.3f}, R={ensemble_recall:.3f}, F1={ensemble_f1:.3f}")

        # 10. 모델 저장
        if ensemble_acc > 0.60:  # 60% 이상만 저장
            model_info = {
                'models': models,
                'scaler': scaler,
                'features': selected_features,
                'ensemble_accuracy': ensemble_acc,
                'best_single_model': best_model[0],
                'timeframe': timeframe,
                'data_size': len(X),
                'trained_at': datetime.now().isoformat()
            }

            filename = f"high_accuracy_{timeframe}_ensemble_model.pkl"
            joblib.dump(model_info, f"models/{filename}")
            print(f"\n  ✅ 모델 저장: models/{filename}")

            return model_info
        else:
            print(f"\n  ⚠️ 정확도 부족으로 저장하지 않음 ({ensemble_acc:.1%})")
            return None

def main():
    trainer = HighAccuracyTrainer()

    print("=" * 60)
    print("🔧 고정확도 모델 훈련 시작")
    print("⏰ 예상 소요 시간: 타임프레임당 10-15분")
    print("=" * 60)

    results = {}

    # 각 타임프레임별 훈련 (데이터 양 조절)
    timeframe_configs = {
        '15m': 20000,  # 약 2달
        '30m': 15000,  # 약 3달
        '1h': 10000,   # 약 1년
        '4h': 5000     # 약 2년
    }

    for timeframe, data_limit in timeframe_configs.items():
        try:
            model_info = trainer.train_ensemble_model(timeframe, data_limit)
            if model_info:
                results[timeframe] = model_info
        except Exception as e:
            print(f"\n❌ {timeframe} 훈련 실패: {e}")

    # 결과 요약
    print("\n" + "=" * 60)
    print("📋 훈련 결과 요약")
    print("=" * 60)

    for tf, info in results.items():
        print(f"\n{tf}:")
        print(f"  앙상블 정확도: {info['ensemble_accuracy']*100:.1f}%")
        print(f"  최고 단일 모델: {info['best_single_model']}")
        print(f"  데이터 크기: {info['data_size']:,}개")

if __name__ == "__main__":
    main()