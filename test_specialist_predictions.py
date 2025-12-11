#!/usr/bin/env python3
"""
전문 모델 예측 테스트 및 시각화
"""

from core.specialist_predictor import SpecialistPredictor
import ccxt
from datetime import datetime

def visualize_probabilities(up_prob, down_prob):
    """확률 시각화"""
    # 막대 그래프 생성
    up_bar = int(up_prob * 50)
    down_bar = int(down_prob * 50)

    up_visual = "█" * up_bar + "░" * (50 - up_bar)
    down_visual = "█" * down_bar + "░" * (50 - down_bar)

    return up_visual, down_visual

def main():
    predictor = SpecialistPredictor()
    exchange = ccxt.binance()

    print("=" * 70)
    print("🎯 BTC 전문 모델 예측 대시보드")
    print("=" * 70)

    # 현재 시장 상황
    ticker = exchange.fetch_ticker('BTC/USDT')
    current_price = ticker['last']
    change_24h = ticker['percentage']

    print(f"\n📊 현재 시장 상황")
    print(f"  BTC 가격: ${current_price:,.0f}")
    print(f"  24시간 변화: {change_24h:+.2f}%")
    print(f"  시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # 모든 타임프레임 예측
    results = predictor.get_all_probabilities()

    print("\n" + "=" * 70)
    print("📈 타임프레임별 상승/하락 확률")
    print("=" * 70)

    for timeframe in ['15m', '30m', '1h', '4h']:
        if timeframe in results:
            data = results[timeframe]
            up_prob = data['up_probability']
            down_prob = data['down_probability']
            signal = data['signal']

            up_visual, down_visual = visualize_probabilities(up_prob, down_prob)

            print(f"\n🕐 {timeframe:4s} 분석")
            print("-" * 60)
            print(f"  상승 📈 {up_prob:5.1%} {up_visual}")
            print(f"  하락 📉 {down_prob:5.1%} {down_visual}")

            # 신호 강도
            if signal == "UP":
                emoji = "🟢"
                strength = up_prob
            elif signal == "DOWN":
                emoji = "🔴"
                strength = down_prob
            else:
                emoji = "⚪"
                strength = max(1 - up_prob - down_prob, 0)

            print(f"  신호: {emoji} {signal:7s} (강도: {strength:.1%})")

    # 종합 분석
    print("\n" + "=" * 70)
    print("📊 종합 분석")
    print("=" * 70)

    # 평균 확률
    avg_up = sum(r['up_probability'] for r in results.values()) / len(results)
    avg_down = sum(r['down_probability'] for r in results.values()) / len(results)

    print(f"\n평균 확률:")
    print(f"  상승: {avg_up:.1%}")
    print(f"  하락: {avg_down:.1%}")

    # 신호 일치도
    signals = [r['signal'] for r in results.values()]
    up_count = signals.count('UP')
    down_count = signals.count('DOWN')
    neutral_count = signals.count('NEUTRAL')

    print(f"\n신호 분포:")
    print(f"  UP: {up_count}개")
    print(f"  DOWN: {down_count}개")
    print(f"  NEUTRAL: {neutral_count}개")

    # 추천 전략
    print(f"\n💡 추천 전략:")

    if avg_up > 0.6:
        print("  ✅ 강한 상승 신호 - 매수 고려")
    elif avg_down > 0.6:
        print("  ❌ 강한 하락 신호 - 매도/관망 고려")
    elif up_count >= 3:
        print("  📈 다수 상승 신호 - 신중한 매수 고려")
    elif down_count >= 3:
        print("  📉 다수 하락 신호 - 신중한 매도 고려")
    else:
        print("  ⚖️ 혼재 신호 - 관망 추천")

    # 리스크 경고
    print(f"\n⚠️ 리스크 경고:")

    # 확률이 너무 낮은 경우
    if avg_up < 0.3 and avg_down < 0.3:
        print("  - 낮은 신뢰도: 모델이 확신하지 못함")

    # 상충되는 신호
    if up_count > 0 and down_count > 0:
        print("  - 상충 신호: 타임프레임별 다른 방향")

    # 극단적 확률
    extreme_up = any(r['up_probability'] > 0.8 for r in results.values())
    extreme_down = any(r['down_probability'] > 0.8 for r in results.values())

    if extreme_up or extreme_down:
        print("  - 극단적 확률: 과신 주의 필요")

    # 최근 예측 기록
    print(f"\n📝 참고사항:")
    print("  - 모든 예측은 확률이며 100% 보장되지 않습니다")
    print("  - 여러 타임프레임을 종합적으로 고려하세요")
    print("  - 리스크 관리를 항상 우선시하세요")

if __name__ == "__main__":
    main()