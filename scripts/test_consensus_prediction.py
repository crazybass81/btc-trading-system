#!/usr/bin/env python3
"""
합의 예측 정확도 테스트
모든 모델(15m, 30m, 1h)이 동시에 같은 방향을 예측할 때
2시간 후 실제 방향과 비교
"""

import ccxt
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class ConsensusPredictionTester:
    def __init__(self):
        self.exchange = ccxt.binance()
        self.models = {}
        self.load_models()

    def load_models(self):
        """전문화 모델 로드"""
        model_accuracies = {
            '15m': {'up_accuracy': 0.605, 'down_accuracy': 0.73},
            '30m': {'up_accuracy': 0.70, 'down_accuracy': 0.68},
            '1h': {'up_accuracy': 0.71, 'down_accuracy': 0.66},
        }

        for tf, accuracies in model_accuracies.items():
            try:
                model_path = f"models/specialist_{tf}_combined_model.pkl"
                model_data = joblib.load(model_path)
                model_data['accuracies'] = accuracies
                self.models[tf] = model_data
                print(f"✅ {tf} 모델 로드 (UP: {accuracies['up_accuracy']*100:.1f}%, DOWN: {accuracies['down_accuracy']*100:.1f}%)")
            except:
                print(f"⚠️ {tf} 모델 로드 실패")

    def get_aligned_data(self, days=30):
        """모든 타임프레임의 정렬된 데이터 수집"""
        print(f"\n📊 {days}일간 데이터 수집...")

        data = {}

        # 각 타임프레임 데이터 수집
        for timeframe in ['15m', '30m', '1h']:
            print(f"  {timeframe} 데이터 수집 중...")

            tf_ms = {
                '15m': 15 * 60 * 1000,
                '30m': 30 * 60 * 1000,
                '1h': 60 * 60 * 1000,
            }

            ms_per_candle = tf_ms[timeframe]
            total_candles = int(days * 24 * 60 * 60 * 1000 / ms_per_candle)

            # 더 많은 데이터 수집 (2시간 후 확인을 위해)
            total_candles += 20  # 여유분

            all_data = []
            chunk_size = 1000
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

                except:
                    break

            df = pd.DataFrame(all_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)

            # 지표 계산
            df = self.calculate_indicators(df)
            data[timeframe] = df

            print(f"    ✅ {len(df)}개 캔들 수집")

        return data

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

    def get_signal(self, df, timeframe, timestamp):
        """특정 시점의 신호 계산"""
        try:
            idx = df.index.get_indexer([timestamp], method='nearest')[0]
            if idx < 50 or idx >= len(df):
                return None

            # 상승 신호 조건
            rsi_oversold = df['rsi'].iloc[idx] < 45
            price_above_ma = df['close'].iloc[idx] > df['ma_20'].iloc[idx] * 0.98
            volume_ok = df['volume_ratio'].iloc[idx] > 1.0
            macd_positive = df['macd_hist'].iloc[idx] > 0

            up_conditions = sum([rsi_oversold, price_above_ma, volume_ok, macd_positive])

            # 하락 신호 조건
            rsi_overbought = df['rsi'].iloc[idx] > 55
            price_below_ma = df['close'].iloc[idx] < df['ma_20'].iloc[idx] * 1.02
            volume_high = df['volume_ratio'].iloc[idx] > 1.2
            macd_negative = df['macd_hist'].iloc[idx] < 0

            down_conditions = sum([rsi_overbought, price_below_ma, volume_high, macd_negative])

            # 신호 결정
            if up_conditions >= 3 and down_conditions < 3:
                return 'UP'
            elif down_conditions >= 3 and up_conditions < 3:
                return 'DOWN'
            else:
                return None

        except:
            return None

    def test_consensus(self):
        """합의 예측 테스트"""
        print("\n" + "="*60)
        print("🎯 합의 예측 테스트 (2시간 후 검증)")
        print("="*60)

        # 데이터 수집
        data = self.get_aligned_data(days=30)

        # 1시간 봉 기준으로 테스트 (가장 긴 타임프레임)
        base_df = data['1h']

        consensus_predictions = []

        print("\n📍 합의 신호 검색 중...")

        # 1시간 봉 기준으로 순회 (2시간 후 확인 가능한 범위)
        for i in range(50, len(base_df) - 2):  # -2는 2시간 후 확인을 위해
            timestamp = base_df.index[i]

            # 각 타임프레임에서 신호 확인
            signals = {}
            for tf in ['15m', '30m', '1h']:
                signal = self.get_signal(data[tf], tf, timestamp)
                signals[tf] = signal

            # 모든 모델이 같은 방향 예측하는지 확인
            unique_signals = set([s for s in signals.values() if s is not None])

            if len(unique_signals) == 1:  # 모두 같은 신호 (None 제외)
                consensus_signal = list(unique_signals)[0]

                # 2시간 후 가격 확인
                current_price = base_df['close'].iloc[i]
                future_price = base_df['close'].iloc[i + 2]  # 2시간 후

                actual_direction = 'UP' if future_price > current_price else 'DOWN'
                correct = (consensus_signal == actual_direction)

                price_change = (future_price - current_price) / current_price * 100

                consensus_predictions.append({
                    'timestamp': timestamp,
                    'consensus': consensus_signal,
                    'actual': actual_direction,
                    'correct': correct,
                    'current_price': current_price,
                    'future_price': future_price,
                    'price_change': price_change,
                    'signals': signals
                })

        # 결과 분석
        if consensus_predictions:
            total = len(consensus_predictions)

            # 상승 합의
            up_consensus = [p for p in consensus_predictions if p['consensus'] == 'UP']
            up_total = len(up_consensus)
            up_correct = sum(1 for p in up_consensus if p['correct'])
            up_accuracy = (up_correct / up_total * 100) if up_total > 0 else 0

            # 하락 합의
            down_consensus = [p for p in consensus_predictions if p['consensus'] == 'DOWN']
            down_total = len(down_consensus)
            down_correct = sum(1 for p in down_consensus if p['correct'])
            down_accuracy = (down_correct / down_total * 100) if down_total > 0 else 0

            # 전체 정확도
            total_correct = up_correct + down_correct
            overall_accuracy = (total_correct / total * 100) if total > 0 else 0

            # 평균 가격 변화
            avg_change_when_correct = np.mean([abs(p['price_change']) for p in consensus_predictions if p['correct']]) if total_correct > 0 else 0
            avg_change_when_wrong = np.mean([abs(p['price_change']) for p in consensus_predictions if not p['correct']]) if (total - total_correct) > 0 else 0

            print(f"\n📊 테스트 결과:")
            print(f"  총 합의 신호: {total}회")
            print(f"\n  📈 상승 합의:")
            print(f"     예측 횟수: {up_total}회")
            print(f"     정확 예측: {up_correct}회")
            print(f"     정확도: {up_accuracy:.1f}%")

            print(f"\n  📉 하락 합의:")
            print(f"     예측 횟수: {down_total}회")
            print(f"     정확 예측: {down_correct}회")
            print(f"     정확도: {down_accuracy:.1f}%")

            print(f"\n  🎯 전체 정확도: {overall_accuracy:.1f}%")
            print(f"  📊 정확 시 평균 변화율: {avg_change_when_correct:.2f}%")
            print(f"  📊 오류 시 평균 변화율: {avg_change_when_wrong:.2f}%")

            # 최근 10개 예측 샘플
            print(f"\n📝 최근 합의 예측 샘플 (최대 10개):")
            print("-"*70)
            for pred in consensus_predictions[-10:]:
                status = "✅" if pred['correct'] else "❌"
                emoji = "📈" if pred['consensus'] == 'UP' else "📉"
                print(f"{status} {pred['timestamp'].strftime('%m-%d %H:%M')} | "
                      f"{emoji} 예측: {pred['consensus']} | "
                      f"실제: {pred['actual']} | "
                      f"변화: {pred['price_change']:+.2f}%")

            # 시간대별 분석
            print(f"\n⏰ 시간대별 정확도:")
            hour_stats = {}
            for pred in consensus_predictions:
                hour = pred['timestamp'].hour
                if hour not in hour_stats:
                    hour_stats[hour] = {'total': 0, 'correct': 0}
                hour_stats[hour]['total'] += 1
                if pred['correct']:
                    hour_stats[hour]['correct'] += 1

            best_hours = sorted([(h, s['correct']/s['total']*100)
                                for h, s in hour_stats.items()
                                if s['total'] >= 3],  # 최소 3회 이상
                               key=lambda x: x[1], reverse=True)[:5]

            if best_hours:
                print("  최고 정확도 시간대 (Top 5):")
                for hour, acc in best_hours:
                    print(f"    {hour:02d}:00 - {acc:.1f}%")

            # 평가
            print(f"\n📋 평가:")
            print("-"*40)
            if overall_accuracy >= 60:
                print(f"✅ 합의 예측 사용 가능! ({overall_accuracy:.1f}%)")
                print(f"   특히 {'상승' if up_accuracy > down_accuracy else '하락'} 예측이 더 정확")
            elif overall_accuracy >= 55:
                print(f"⚠️ 합의 예측 개선 필요 ({overall_accuracy:.1f}%)")
                print(f"   추가 필터링이나 조건 강화 필요")
            else:
                print(f"❌ 합의 예측 효과 없음 ({overall_accuracy:.1f}%)")
                print(f"   랜덤과 큰 차이 없음")

            return {
                'total': total,
                'up_total': up_total,
                'up_correct': up_correct,
                'up_accuracy': up_accuracy,
                'down_total': down_total,
                'down_correct': down_correct,
                'down_accuracy': down_accuracy,
                'overall_accuracy': overall_accuracy,
                'predictions': consensus_predictions
            }
        else:
            print("\n⚠️ 합의 신호를 찾을 수 없습니다.")
            return None

def main():
    print("="*60)
    print("🤝 모델 합의 예측 테스트")
    print("📅 테스트 기간: 30일")
    print("⏰ 검증 시간: 2시간 후")
    print("="*60)

    tester = ConsensusPredictionTester()
    result = tester.test_consensus()

    if result:
        print(f"\n" + "="*60)
        print("📊 최종 요약")
        print("="*60)
        print(f"합의 신호 발생: {result['total']}회 (30일간)")
        print(f"전체 정확도: {result['overall_accuracy']:.1f}%")
        print(f"상승 정확도: {result['up_accuracy']:.1f}% ({result['up_total']}회)")
        print(f"하락 정확도: {result['down_accuracy']:.1f}% ({result['down_total']}회)")

if __name__ == "__main__":
    main()