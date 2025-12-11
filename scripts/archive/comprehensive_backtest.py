#!/usr/bin/env python3
"""
Comprehensive Backtest for All BTC Direction Prediction Models
긴 기간 데이터를 이용한 모든 모델의 종합 백테스트
"""

import numpy as np
import pandas as pd
import joblib
import ccxt
from datetime import datetime, timedelta
import warnings
import json
import os
warnings.filterwarnings('ignore')

# Exchange 설정
exchange = ccxt.binance({
    'enableRateLimit': True,
    'options': {'defaultType': 'future'}
})

def collect_extended_data(symbol='BTC/USDT', timeframe='15m', days=120):
    """더 긴 기간의 데이터 수집 (기본 120일)"""
    print(f"\n📊 Collecting {days} days of {timeframe} data...")

    limit = 1500
    ohlcv_list = []

    # timeframe을 밀리초로 변환
    timeframe_ms = {
        '15m': 15 * 60 * 1000,
        '30m': 30 * 60 * 1000,
        '1h': 60 * 60 * 1000,
        '4h': 4 * 60 * 60 * 1000
    }[timeframe]

    # 시작 시간 계산
    end_time = exchange.milliseconds()
    start_time = end_time - (days * 24 * 60 * 60 * 1000)
    current_time = start_time

    while current_time < end_time:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=current_time, limit=limit)
            if not ohlcv:
                break
            ohlcv_list.extend(ohlcv)
            current_time = ohlcv[-1][0] + timeframe_ms
            if len(ohlcv) < limit:
                break
        except Exception as e:
            print(f"Error fetching data: {e}")
            break

    # DataFrame 생성
    df = pd.DataFrame(ohlcv_list, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df = df.drop_duplicates(subset=['timestamp']).sort_values('timestamp').reset_index(drop=True)

    print(f"✅ Collected {len(df)} candles from {df['timestamp'].min()} to {df['timestamp'].max()}")
    return df

def prepare_features(df):
    """특징 생성"""
    # 기본 특징
    df['returns'] = df['close'].pct_change()
    df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
    df['volume_ratio'] = df['volume'] / df['volume'].rolling(window=20).mean()

    # 변동성
    df['volatility'] = df['returns'].rolling(window=20).std()
    df['high_low_ratio'] = (df['high'] - df['low']) / df['close']
    df['close_open_ratio'] = (df['close'] - df['open']) / df['open']

    # 기술적 지표
    for period in [5, 10, 20, 50]:
        df[f'sma_{period}'] = df['close'].rolling(window=period).mean()
        df[f'sma_ratio_{period}'] = df['close'] / df[f'sma_{period}']

    for period in [7, 14, 21]:
        df[f'rsi_{period}'] = calculate_rsi(df['close'], period)

    # MACD
    exp1 = df['close'].ewm(span=12, adjust=False).mean()
    exp2 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = exp1 - exp2
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_diff'] = df['macd'] - df['macd_signal']

    # 볼린저 밴드
    for period in [20, 30]:
        sma = df['close'].rolling(window=period).mean()
        std = df['close'].rolling(window=period).std()
        df[f'bb_upper_{period}'] = sma + (std * 2)
        df[f'bb_lower_{period}'] = sma - (std * 2)
        df[f'bb_ratio_{period}'] = (df['close'] - df[f'bb_lower_{period}']) / (df[f'bb_upper_{period}'] - df[f'bb_lower_{period}'])

    # 시간 특징
    df['hour'] = df['timestamp'].dt.hour
    df['dayofweek'] = df['timestamp'].dt.dayofweek
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['day_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
    df['day_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)

    return df

def calculate_rsi(prices, period=14):
    """RSI 계산"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def backtest_model(model_path, df, timeframe, direction, days_for_test=90):
    """단일 모델 백테스트"""
    model_name = os.path.basename(model_path).replace('.pkl', '')
    print(f"\n🔍 Backtesting {model_name}...")

    try:
        # 모델 로드
        model = joblib.load(model_path)

        # 타겟 생성 (다음 봉 방향)
        df['next_direction'] = (df['close'].shift(-1) > df['close']).astype(int)

        # 특징 준비
        feature_cols = [col for col in df.columns if col not in
                       ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'next_direction']]

        # NaN 제거
        df_clean = df.dropna()

        # 테스트 데이터 분리 (마지막 90일)
        split_date = df_clean['timestamp'].max() - timedelta(days=days_for_test)
        test_df = df_clean[df_clean['timestamp'] > split_date].copy()

        if len(test_df) < 100:
            print(f"❌ Insufficient test data for {model_name}")
            return None

        X_test = test_df[feature_cols]
        y_test = test_df['next_direction']

        # 예측
        if direction.lower() == 'up':
            # UP 모델: 상승 예측
            y_pred = model.predict(X_test)
            predictions_correct = (y_pred == 1) & (y_test == 1)  # 상승 예측이 맞은 경우
            predictions_made = (y_pred == 1)  # 상승 예측한 모든 경우
        else:
            # DOWN 모델: 하락 예측
            y_pred = model.predict(X_test)
            predictions_correct = (y_pred == 0) & (y_test == 0)  # 하락 예측이 맞은 경우
            predictions_made = (y_pred == 0)  # 하락 예측한 모든 경우

        # 정확도 계산
        if predictions_made.sum() > 0:
            accuracy = predictions_correct.sum() / predictions_made.sum()
        else:
            accuracy = 0

        # 상세 결과
        results = {
            'model': model_name,
            'timeframe': timeframe,
            'direction': direction.upper(),
            'test_days': days_for_test,
            'test_samples': len(test_df),
            'predictions_made': int(predictions_made.sum()),
            'correct_predictions': int(predictions_correct.sum()),
            'accuracy': float(accuracy),
            'test_period': f"{test_df['timestamp'].min()} to {test_df['timestamp'].max()}",

            # 추가 통계
            'daily_predictions': predictions_made.sum() / days_for_test,
            'win_rate': float(accuracy * 100),

            # 시간대별 성능
            'performance_by_hour': {},
            'performance_by_day': {}
        }

        # 시간대별 분석
        test_df['prediction'] = y_pred
        test_df['correct'] = predictions_correct

        for hour in range(24):
            hour_data = test_df[test_df['hour'] == hour]
            if len(hour_data) > 0:
                if direction.lower() == 'up':
                    hour_preds = (hour_data['prediction'] == 1).sum()
                    hour_correct = ((hour_data['prediction'] == 1) & (hour_data['next_direction'] == 1)).sum()
                else:
                    hour_preds = (hour_data['prediction'] == 0).sum()
                    hour_correct = ((hour_data['prediction'] == 0) & (hour_data['next_direction'] == 0)).sum()

                if hour_preds > 0:
                    results['performance_by_hour'][f"{hour:02d}:00"] = {
                        'accuracy': float(hour_correct / hour_preds),
                        'predictions': int(hour_preds)
                    }

        # 요일별 분석
        for day in range(7):
            day_data = test_df[test_df['dayofweek'] == day]
            if len(day_data) > 0:
                if direction.lower() == 'up':
                    day_preds = (day_data['prediction'] == 1).sum()
                    day_correct = ((day_data['prediction'] == 1) & (day_data['next_direction'] == 1)).sum()
                else:
                    day_preds = (day_data['prediction'] == 0).sum()
                    day_correct = ((day_data['prediction'] == 0) & (day_data['next_direction'] == 0)).sum()

                if day_preds > 0:
                    days_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
                    results['performance_by_day'][days_names[day]] = {
                        'accuracy': float(day_correct / day_preds),
                        'predictions': int(day_preds)
                    }

        print(f"✅ {model_name}: {accuracy*100:.1f}% accuracy on {predictions_made.sum()} predictions")
        return results

    except Exception as e:
        print(f"❌ Error backtesting {model_name}: {e}")
        return None

def main():
    """메인 실행 함수"""
    print("=" * 60)
    print("🚀 COMPREHENSIVE MODEL BACKTEST - EXTENDED PERIOD")
    print("=" * 60)

    # 모델 목록
    models_to_test = [
        ('models/deep_ensemble_1h_up_model.pkl', '1h', 'up'),
        ('models/deep_ensemble_1h_down_model.pkl', '1h', 'down'),
        ('models/deep_ensemble_4h_up_model.pkl', '4h', 'up'),
        ('models/deep_ensemble_4h_down_model.pkl', '4h', 'down'),
        ('models/deep_ensemble_30m_up_model.pkl', '30m', 'up'),
        ('models/deep_ensemble_30m_down_model.pkl', '30m', 'down'),
        ('models/advanced_15m_up_model.pkl', '15m', 'up'),
        ('models/deep_ensemble_15m_up_model.pkl', '15m', 'up')
    ]

    all_results = []

    # 각 시간봉별 데이터 수집 및 백테스트
    for timeframe in ['15m', '30m', '1h', '4h']:
        print(f"\n📊 Processing {timeframe} timeframe...")

        # 120일 데이터 수집
        df = collect_extended_data(timeframe=timeframe, days=120)

        # 특징 생성
        df = prepare_features(df)

        # 해당 시간봉 모델들 테스트
        for model_path, tf, direction in models_to_test:
            if tf == timeframe:
                result = backtest_model(model_path, df, timeframe, direction, days_for_test=90)
                if result:
                    all_results.append(result)

    # 결과 저장
    results_df = pd.DataFrame(all_results)
    results_df.to_csv('backtest_results_extended.csv', index=False)

    # JSON으로도 저장
    with open('backtest_results_extended.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    # 결과 요약
    print("\n" + "=" * 60)
    print("📊 BACKTEST SUMMARY - 90 DAYS TEST PERIOD")
    print("=" * 60)

    for result in all_results:
        print(f"\n🎯 {result['model']}")
        print(f"   Timeframe: {result['timeframe']}")
        print(f"   Direction: {result['direction']}")
        print(f"   Accuracy: {result['accuracy']*100:.2f}%")
        print(f"   Total Predictions: {result['predictions_made']}")
        print(f"   Correct Predictions: {result['correct_predictions']}")
        print(f"   Test Period: {result['test_period']}")

        # 최고 성능 시간대
        if result['performance_by_hour']:
            best_hour = max(result['performance_by_hour'].items(),
                          key=lambda x: x[1]['accuracy'])
            print(f"   Best Hour: {best_hour[0]} ({best_hour[1]['accuracy']*100:.1f}%)")

    # 전체 평균
    avg_accuracy = results_df['accuracy'].mean()
    print(f"\n{'='*60}")
    print(f"📈 OVERALL AVERAGE ACCURACY: {avg_accuracy*100:.2f}%")
    print(f"📅 Test Period: 90 days of real market data")
    print(f"📊 Total Models Tested: {len(all_results)}")
    print(f"{'='*60}")

    return results_df

if __name__ == "__main__":
    results = main()