#!/usr/bin/env python3
"""
Comprehensive Backtest for All BTC Direction Prediction Models V2
딕셔너리 형태의 앙상블 모델을 처리하는 개선된 버전
"""

import numpy as np
import pandas as pd
import joblib
import ccxt
from datetime import datetime, timedelta
import warnings
import json
import os
from sklearn.preprocessing import RobustScaler
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

def predict_ensemble(model_data, X_test):
    """앙상블 모델 예측 함수"""
    models = model_data['models']
    weights = model_data.get('weights', [1/len(models)] * len(models))
    scaler = model_data.get('scaler')

    # 특징 스케일링
    if scaler:
        X_test_scaled = scaler.transform(X_test)
    else:
        X_test_scaled = X_test

    # 각 모델의 예측 수집
    predictions = []
    for model in models:
        try:
            pred = model.predict_proba(X_test_scaled)[:, 1]
            predictions.append(pred)
        except:
            pred = model.predict(X_test_scaled)
            predictions.append(pred)

    # 가중 평균
    weighted_predictions = np.average(predictions, axis=0, weights=weights)

    # 이진 분류 (0.5 기준)
    return (weighted_predictions > 0.5).astype(int)

def backtest_model(model_path, df, timeframe, direction, days_for_test=90):
    """단일 모델 백테스트"""
    model_name = os.path.basename(model_path).replace('.pkl', '')
    print(f"\n🔍 Backtesting {model_name}...")

    try:
        # 모델 로드
        model_data = joblib.load(model_path)

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

        # 앙상블 예측
        if isinstance(model_data, dict):
            # 딕셔너리 형태의 앙상블 모델
            y_pred = predict_ensemble(model_data, X_test)
        else:
            # 단일 모델
            y_pred = model_data.predict(X_test)

        # 정확도 계산
        if direction.lower() == 'up':
            # UP 모델: 상승 예측
            predictions_correct = (y_pred == 1) & (y_test == 1)  # 상승 예측이 맞은 경우
            predictions_made = (y_pred == 1)  # 상승 예측한 모든 경우
        else:
            # DOWN 모델: 하락 예측
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
            'performance_by_day': {},

            # 월별 성능
            'monthly_performance': {}
        }

        # 시간대별 분석
        test_df['prediction'] = y_pred
        test_df['correct'] = predictions_correct

        # 시간별 성능
        for hour in range(0, 24, 3):  # 3시간 단위로
            hour_data = test_df[test_df['hour'].between(hour, hour+2)]
            if len(hour_data) > 10:
                if direction.lower() == 'up':
                    hour_preds = (hour_data['prediction'] == 1).sum()
                    hour_correct = ((hour_data['prediction'] == 1) & (hour_data['next_direction'] == 1)).sum()
                else:
                    hour_preds = (hour_data['prediction'] == 0).sum()
                    hour_correct = ((hour_data['prediction'] == 0) & (hour_data['next_direction'] == 0)).sum()

                if hour_preds > 0:
                    results['performance_by_hour'][f"{hour:02d}:00-{hour+2:02d}:00"] = {
                        'accuracy': float(hour_correct / hour_preds),
                        'predictions': int(hour_preds)
                    }

        # 요일별 분석
        days_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        for day in range(7):
            day_data = test_df[test_df['dayofweek'] == day]
            if len(day_data) > 10:
                if direction.lower() == 'up':
                    day_preds = (day_data['prediction'] == 1).sum()
                    day_correct = ((day_data['prediction'] == 1) & (day_data['next_direction'] == 1)).sum()
                else:
                    day_preds = (day_data['prediction'] == 0).sum()
                    day_correct = ((day_data['prediction'] == 0) & (day_data['next_direction'] == 0)).sum()

                if day_preds > 0:
                    results['performance_by_day'][days_names[day]] = {
                        'accuracy': float(day_correct / day_preds),
                        'predictions': int(day_preds)
                    }

        # 월별 성능 분석
        test_df['month'] = test_df['timestamp'].dt.to_period('M')
        for month in test_df['month'].unique():
            month_data = test_df[test_df['month'] == month]
            if len(month_data) > 10:
                if direction.lower() == 'up':
                    month_preds = (month_data['prediction'] == 1).sum()
                    month_correct = ((month_data['prediction'] == 1) & (month_data['next_direction'] == 1)).sum()
                else:
                    month_preds = (month_data['prediction'] == 0).sum()
                    month_correct = ((month_data['prediction'] == 0) & (month_data['next_direction'] == 0)).sum()

                if month_preds > 0:
                    results['monthly_performance'][str(month)] = {
                        'accuracy': float(month_correct / month_preds),
                        'predictions': int(month_preds),
                        'samples': len(month_data)
                    }

        print(f"✅ {model_name}: {accuracy*100:.1f}% accuracy on {predictions_made.sum()} predictions")
        print(f"   Test period: {test_df['timestamp'].min().date()} to {test_df['timestamp'].max().date()}")

        # 최고/최저 성능 시간대 출력
        if results['performance_by_hour']:
            best_hour = max(results['performance_by_hour'].items(), key=lambda x: x[1]['accuracy'])
            worst_hour = min(results['performance_by_hour'].items(), key=lambda x: x[1]['accuracy'])
            print(f"   Best time: {best_hour[0]} ({best_hour[1]['accuracy']*100:.1f}%)")
            print(f"   Worst time: {worst_hour[0]} ({worst_hour[1]['accuracy']*100:.1f}%)")

        return results

    except Exception as e:
        print(f"❌ Error backtesting {model_name}: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """메인 실행 함수"""
    print("=" * 70)
    print("🚀 COMPREHENSIVE MODEL BACKTEST - EXTENDED PERIOD (120 DAYS)")
    print("=" * 70)

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
        print(f"\n{'='*70}")
        print(f"📊 Processing {timeframe} timeframe...")
        print(f"{'='*70}")

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
    if all_results:
        results_df = pd.DataFrame(all_results)
        results_df.to_csv('backtest_results_extended.csv', index=False)

        # JSON으로도 저장
        with open('backtest_results_extended.json', 'w') as f:
            json.dump(all_results, f, indent=2, default=str)

        # 결과 요약
        print("\n" + "=" * 70)
        print("📊 BACKTEST SUMMARY - 90 DAYS TEST PERIOD")
        print("=" * 70)

        for result in all_results:
            print(f"\n🎯 {result['model']}")
            print(f"   Timeframe: {result['timeframe']}")
            print(f"   Direction: {result['direction']}")
            print(f"   Accuracy: {result['accuracy']*100:.2f}%")
            print(f"   Total Predictions: {result['predictions_made']}")
            print(f"   Correct Predictions: {result['correct_predictions']}")
            print(f"   Daily Average: {result['daily_predictions']:.1f} predictions/day")

            # 월별 성능
            if result.get('monthly_performance'):
                print(f"   Monthly Performance:")
                for month, perf in sorted(result['monthly_performance'].items()):
                    print(f"      {month}: {perf['accuracy']*100:.1f}% ({perf['predictions']} predictions)")

        # 전체 평균
        avg_accuracy = results_df['accuracy'].mean()
        total_predictions = results_df['predictions_made'].sum()
        total_correct = results_df['correct_predictions'].sum()

        print(f"\n{'='*70}")
        print(f"📈 OVERALL STATISTICS")
        print(f"{'='*70}")
        print(f"Average Accuracy: {avg_accuracy*100:.2f}%")
        print(f"Total Predictions: {total_predictions}")
        print(f"Total Correct: {total_correct}")
        print(f"Overall Win Rate: {total_correct/total_predictions*100:.2f}%")
        print(f"Test Period: 90 days of real market data")
        print(f"Total Models Tested: {len(all_results)}")
        print(f"{'='*70}")

        # 최고/최저 성능 모델
        best_model = results_df.loc[results_df['accuracy'].idxmax()]
        worst_model = results_df.loc[results_df['accuracy'].idxmin()]

        print(f"\n🏆 Best Model: {best_model['model']}")
        print(f"   Accuracy: {best_model['accuracy']*100:.2f}%")
        print(f"   Direction: {best_model['direction']}")

        print(f"\n⚠️ Weakest Model: {worst_model['model']}")
        print(f"   Accuracy: {worst_model['accuracy']*100:.2f}%")
        print(f"   Direction: {worst_model['direction']}")

        return results_df
    else:
        print("\n❌ No successful backtests completed")
        return None

if __name__ == "__main__":
    results = main()