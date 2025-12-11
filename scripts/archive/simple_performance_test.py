#!/usr/bin/env python3
"""
간단한 성능 테스트
훈련 정확도 vs 실제 정확도 비교
"""

import ccxt
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def test_direction_accuracy():
    """단순 방향 예측 정확도 테스트"""
    print("="*60)
    print("📊 BTC 15분 방향 예측 통계")
    print("📅 " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("="*60)

    # Get data
    exchange = ccxt.binance()
    ohlcv = exchange.fetch_ohlcv('BTC/USDT', '15m', limit=1000)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)

    print(f"\n📊 데이터 분석 ({len(df)}개 캔들)")
    print(f"  기간: {df.index[0].strftime('%Y-%m-%d')} ~ {df.index[-1].strftime('%Y-%m-%d')}")
    print(f"  현재가: ${df['close'].iloc[-1]:,.2f}")

    # Calculate movements
    movements = []
    for i in range(len(df) - 1):
        movement = 1 if df['close'].iloc[i+1] > df['close'].iloc[i] else 0
        movements.append(movement)

    up_count = sum(movements)
    down_count = len(movements) - up_count

    print(f"\n📈 실제 시장 움직임:")
    print(f"  상승: {up_count}/{len(movements)} ({up_count/len(movements)*100:.1f}%)")
    print(f"  하락: {down_count}/{len(movements)} ({down_count/len(movements)*100:.1f}%)")

    # Strategy comparisons
    print(f"\n🎯 전략별 예상 정확도:")

    # 1. Random guess
    print(f"  1. 무작위 예측: ~50%")

    # 2. Always UP
    always_up_accuracy = up_count / len(movements) * 100
    print(f"  2. 항상 UP 예측: {always_up_accuracy:.1f}%")

    # 3. Always DOWN
    always_down_accuracy = down_count / len(movements) * 100
    print(f"  3. 항상 DOWN 예측: {always_down_accuracy:.1f}%")

    # 4. Trend following (previous candle)
    trend_correct = 0
    for i in range(1, len(movements)):
        if movements[i] == movements[i-1]:
            trend_correct += 1
    trend_accuracy = trend_correct / (len(movements)-1) * 100
    print(f"  4. 추세 추종: {trend_accuracy:.1f}%")

    # 5. Mean reversion
    revert_correct = 0
    for i in range(1, len(movements)):
        if movements[i] != movements[i-1]:
            revert_correct += 1
    revert_accuracy = revert_correct / (len(movements)-1) * 100
    print(f"  5. 평균 회귀: {revert_accuracy:.1f}%")

    # Pattern analysis
    print(f"\n📊 패턴 분석:")

    # Consecutive ups/downs
    max_consecutive_up = 0
    max_consecutive_down = 0
    current_up = 0
    current_down = 0

    for m in movements:
        if m == 1:
            current_up += 1
            current_down = 0
            max_consecutive_up = max(max_consecutive_up, current_up)
        else:
            current_down += 1
            current_up = 0
            max_consecutive_down = max(max_consecutive_down, current_down)

    print(f"  최대 연속 상승: {max_consecutive_up}개")
    print(f"  최대 연속 하락: {max_consecutive_down}개")

    # Volatility periods
    volatility = df['close'].pct_change().rolling(20).std()
    high_vol_periods = (volatility > volatility.median() * 1.5).sum()
    print(f"  고변동성 기간: {high_vol_periods}/{len(volatility)} ({high_vol_periods/len(volatility)*100:.1f}%)")

    # Time analysis
    hourly_up = {}
    for i, time in enumerate(df.index[:-1]):
        hour = time.hour
        if hour not in hourly_up:
            hourly_up[hour] = {'up': 0, 'total': 0}
        hourly_up[hour]['total'] += 1
        if movements[i] == 1:
            hourly_up[hour]['up'] += 1

    print(f"\n⏰ 시간대별 상승 확률:")
    best_hours = []
    for hour in sorted(hourly_up.keys()):
        if hourly_up[hour]['total'] >= 10:  # 충분한 샘플
            up_rate = hourly_up[hour]['up'] / hourly_up[hour]['total'] * 100
            if up_rate >= 55:  # 55% 이상
                best_hours.append((hour, up_rate))
                print(f"  {hour:02d}:00 UTC: {up_rate:.1f}%")

    if not best_hours:
        print("  특별한 시간대 없음")

    # Model performance context
    print(f"\n🎯 ML 모델 목표:")
    print(f"  ❌ 50% 이하: 무작위 수준")
    print(f"  ⚠️ 50-55%: 약간의 예측력")
    print(f"  ✅ 55-60%: 유의미한 예측력")
    print(f"  🎯 60% 이상: 우수한 예측력")

    print(f"\n📋 우리 모델 성과:")
    print(f"  ✅ Deep Ensemble 15m UP: 62.8% (훈련)")
    print(f"  ✅ Advanced ML 15m UP: 65.2% (훈련)")
    print(f"  💡 실제 백테스트 필요")

    # Trading implications
    print(f"\n💰 거래 시뮬레이션 (60% 정확도 가정):")
    simulated_trades = 100
    win_rate = 0.6
    avg_win = 0.5  # 0.5% per trade
    avg_loss = 0.5
    expected_return = (win_rate * avg_win - (1-win_rate) * avg_loss) * simulated_trades
    print(f"  100회 거래시 기대 수익: {expected_return:.1f}%")

    if win_rate > 0.55:
        print(f"  ✅ 수익 가능성 있음")
    else:
        print(f"  ❌ 수익 어려움")

if __name__ == "__main__":
    test_direction_accuracy()