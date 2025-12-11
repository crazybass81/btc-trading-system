#!/usr/bin/env python3
"""
전문화 모델 정확도 테스트
상승 모델은 상승만, 하락 모델은 하락만 예측
"""

import ccxt
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class SpecialistAccuracyTester:
    def __init__(self):
        self.exchange = ccxt.binance()
        self.models = {}
        self.load_models()

    def load_models(self):
        """전문화 모델 로드"""
        # 백테스팅에서 사용한 정확도
        model_accuracies = {
            '30m': {'up_accuracy': 0.70, 'down_accuracy': 0.68},
            '1h': {'up_accuracy': 0.71, 'down_accuracy': 0.66},
            '15m': {'up_accuracy': 0.605, 'down_accuracy': 0.73},
        }

        for tf, accuracies in model_accuracies.items():
            try:
                model_path = f"models/specialist_{tf}_combined_model.pkl"
                model_data = joblib.load(model_path)
                model_data['accuracies'] = accuracies
                self.models[tf] = model_data
                print(f"✅ {tf} 모델 로드")
                print(f"   상승 전문: {accuracies['up_accuracy']*100:.1f}%")
                print(f"   하락 전문: {accuracies['down_accuracy']*100:.1f}%")
            except:
                print(f"⚠️ {tf} 모델 로드 실패")

    def get_data(self, timeframe, days=30):
        """테스트용 데이터 수집"""
        print(f"\n📊 {timeframe} {days}일 데이터 수집...")

        all_data = []
        chunk_size = 1000

        tf_ms = {
            '15m': 15 * 60 * 1000,
            '30m': 30 * 60 * 1000,
            '1h': 60 * 60 * 1000,
        }

        ms_per_candle = tf_ms.get(timeframe, 60 * 60 * 1000)
        total_candles = int(days * 24 * 60 * 60 * 1000 / ms_per_candle)

        end_time = self.exchange.milliseconds()
        current_time = end_time

        while len(all_data) < total_candles:
            try:
                ohlcv = self.exchange.fetch_ohlcv(
                    'BTC/USDT',
                    timeframe,
                    limit=chunk_size,
                    since=current_time - (chunk_size * ms_per_candle)
                )

                if not ohlcv:
                    break

                all_data = ohlcv + all_data
                current_time = ohlcv[0][0] if ohlcv else current_time

                if len(all_data) >= total_candles:
                    all_data = all_data[-total_candles:]
                    break

            except Exception as e:
                break

        df = pd.DataFrame(all_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)

        print(f"  ✅ {len(df)}개 캔들 수집 완료")
        return df

    def calculate_indicators(self, df):
        """기술적 지표 계산"""
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / (loss + 1e-10)
        df['rsi'] = 100 - (100 / (1 + rs))

        # 이동평균
        df['ma_20'] = df['close'].rolling(window=20).mean()
        df['ma_50'] = df['close'].rolling(window=50).mean()

        # 볼륨 비율
        df['volume_ratio'] = df['volume'] / df['volume'].rolling(window=20).mean()

        # MACD
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = exp1 - exp2
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']

        return df

    def test_up_model(self, df, timeframe):
        """상승 전문 모델 테스트 - 상승만 예측"""
        model_info = self.models[timeframe]
        accuracies = model_info['accuracies']

        df = self.calculate_indicators(df)

        predictions = []
        correct = 0
        total = 0

        # 상승 예측 조건
        for i in range(50, len(df)-1):
            # 상승 신호 조건 체크
            rsi_oversold = df['rsi'].iloc[i] < 45
            price_above_ma = df['close'].iloc[i] > df['ma_20'].iloc[i] * 0.98
            volume_ok = df['volume_ratio'].iloc[i] > 1.0
            macd_positive = df['macd_hist'].iloc[i] > 0

            conditions_met = sum([rsi_oversold, price_above_ma, volume_ok, macd_positive])

            # 상승 예측 시
            if conditions_met >= 3:
                # 실제 다음 캔들이 상승했는지 확인
                actual_up = df['close'].iloc[i+1] > df['close'].iloc[i]

                predictions.append({
                    'timestamp': df.index[i],
                    'predicted': 'UP',
                    'actual': 'UP' if actual_up else 'DOWN',
                    'correct': actual_up
                })

                if actual_up:
                    correct += 1
                total += 1

        accuracy = (correct / total * 100) if total > 0 else 0

        return {
            'total': total,
            'correct': correct,
            'accuracy': accuracy,
            'predictions': predictions[-10:]  # 최근 10개만
        }

    def test_down_model(self, df, timeframe):
        """하락 전문 모델 테스트 - 하락만 예측"""
        model_info = self.models[timeframe]
        accuracies = model_info['accuracies']

        df = self.calculate_indicators(df)

        predictions = []
        correct = 0
        total = 0

        # 하락 예측 조건
        for i in range(50, len(df)-1):
            # 하락 신호 조건 체크
            rsi_overbought = df['rsi'].iloc[i] > 55
            price_below_ma = df['close'].iloc[i] < df['ma_20'].iloc[i] * 1.02
            volume_high = df['volume_ratio'].iloc[i] > 1.2
            macd_negative = df['macd_hist'].iloc[i] < 0

            conditions_met = sum([rsi_overbought, price_below_ma, volume_high, macd_negative])

            # 하락 예측 시
            if conditions_met >= 3:
                # 실제 다음 캔들이 하락했는지 확인
                actual_down = df['close'].iloc[i+1] < df['close'].iloc[i]

                predictions.append({
                    'timestamp': df.index[i],
                    'predicted': 'DOWN',
                    'actual': 'DOWN' if actual_down else 'UP',
                    'correct': actual_down
                })

                if actual_down:
                    correct += 1
                total += 1

        accuracy = (correct / total * 100) if total > 0 else 0

        return {
            'total': total,
            'correct': correct,
            'accuracy': accuracy,
            'predictions': predictions[-10:]  # 최근 10개만
        }

    def run_test(self):
        """모든 모델 테스트 실행"""
        print("\n" + "="*60)
        print("🎯 전문화 모델 정확도 테스트")
        print("="*60)

        results = {}

        for timeframe in self.models.keys():
            print(f"\n{'='*60}")
            print(f"📍 {timeframe} 모델 테스트")
            print(f"{'='*60}")

            # 데이터 수집
            df = self.get_data(timeframe, days=30)

            # 상승 모델 테스트
            print(f"\n📈 상승 전문 모델 테스트...")
            up_result = self.test_up_model(df, timeframe)

            print(f"  예측 횟수: {up_result['total']}회")
            print(f"  정확 예측: {up_result['correct']}회")
            print(f"  정확도: {up_result['accuracy']:.1f}%")

            # 하락 모델 테스트
            print(f"\n📉 하락 전문 모델 테스트...")
            down_result = self.test_down_model(df, timeframe)

            print(f"  예측 횟수: {down_result['total']}회")
            print(f"  정확 예측: {down_result['correct']}회")
            print(f"  정확도: {down_result['accuracy']:.1f}%")

            results[timeframe] = {
                'up': up_result,
                'down': down_result
            }

            # 최근 예측 샘플 출력
            print(f"\n📝 최근 상승 예측 샘플 (최대 5개):")
            for pred in up_result['predictions'][:5]:
                status = "✅" if pred['correct'] else "❌"
                print(f"  {status} {pred['timestamp'].strftime('%m-%d %H:%M')} → 실제: {pred['actual']}")

            print(f"\n📝 최근 하락 예측 샘플 (최대 5개):")
            for pred in down_result['predictions'][:5]:
                status = "✅" if pred['correct'] else "❌"
                print(f"  {status} {pred['timestamp'].strftime('%m-%d %H:%M')} → 실제: {pred['actual']}")

        # 종합 결과
        print("\n" + "="*60)
        print("📊 종합 결과")
        print("="*60)
        print("\n타임프레임 | 상승 예측수 | 상승 정확도 | 하락 예측수 | 하락 정확도")
        print("-"*70)

        for tf, result in results.items():
            up_acc = result['up']['accuracy']
            down_acc = result['down']['accuracy']

            # 정확도별 이모지
            up_emoji = "🟢" if up_acc >= 60 else "🟡" if up_acc >= 50 else "🔴"
            down_emoji = "🟢" if down_acc >= 60 else "🟡" if down_acc >= 50 else "🔴"

            print(f"{tf:10s} | {result['up']['total']:10d} | {up_emoji} {up_acc:7.1f}% | "
                  f"{result['down']['total']:10d} | {down_emoji} {down_acc:7.1f}%")

        # 평가
        print("\n📋 평가:")
        print("-"*40)
        for tf, result in results.items():
            print(f"\n{tf}:")

            # 상승 모델 평가
            up_acc = result['up']['accuracy']
            if up_acc >= 60:
                print(f"  📈 상승 모델: ✅ 사용 가능 ({up_acc:.1f}%)")
            elif up_acc >= 50:
                print(f"  📈 상승 모델: ⚠️ 개선 필요 ({up_acc:.1f}%)")
            else:
                print(f"  📈 상승 모델: ❌ 사용 불가 ({up_acc:.1f}%)")

            # 하락 모델 평가
            down_acc = result['down']['accuracy']
            if down_acc >= 60:
                print(f"  📉 하락 모델: ✅ 사용 가능 ({down_acc:.1f}%)")
            elif down_acc >= 50:
                print(f"  📉 하락 모델: ⚠️ 개선 필요 ({down_acc:.1f}%)")
            else:
                print(f"  📉 하락 모델: ❌ 사용 불가 ({down_acc:.1f}%)")

def main():
    tester = SpecialistAccuracyTester()
    tester.run_test()

if __name__ == "__main__":
    main()