#!/usr/bin/env python3
"""
균형 모델 통합 테스트
"""

from core.main import BTCTradingSystem
import ccxt
import pandas as pd
from datetime import datetime

def test_balanced_models():
    """균형 모델 예측 테스트"""
    ts = BTCTradingSystem()
    exchange = ccxt.binance()

    print("=" * 60)
    print("🔧 균형 모델 예측 테스트")
    print("=" * 60)

    # 현재 시장 상황 확인
    ticker = exchange.fetch_ticker('BTC/USDT')
    current_price = ticker['last']
    change_24h = ticker['percentage']

    print(f"\n📊 현재 시장 상황:")
    print(f"  BTC 가격: ${current_price:,.0f}")
    print(f"  24시간 변화: {change_24h:+.2f}%")

    # 각 타임프레임별 예측
    print("\n🔮 모델 예측 결과:")
    print("-" * 40)

    predictions = {}
    for timeframe in ['15m', '30m', '1h', '4h']:
        signal, confidence = ts.get_ml_prediction(timeframe)
        predictions[timeframe] = (signal, confidence)

        # 이모지 설정
        emoji = "📈" if signal == "UP" else "📉"

        print(f"{timeframe:4s}: {emoji} {signal:4s} (신뢰도: {confidence:.1f}%)")

    # 예측 일관성 분석
    print("\n📊 예측 분석:")
    print("-" * 40)

    up_count = sum(1 for s, _ in predictions.values() if s == "UP")
    down_count = sum(1 for s, _ in predictions.values() if s == "DOWN")

    print(f"UP 예측: {up_count}개")
    print(f"DOWN 예측: {down_count}개")

    # 종합 신호
    if up_count > down_count:
        overall = "BULLISH 📈"
    elif down_count > up_count:
        overall = "BEARISH 📉"
    else:
        overall = "NEUTRAL ⚖️"

    print(f"\n종합 전망: {overall}")

    # 최근 예측 기록 확인 (실제 가격 움직임과 비교)
    print("\n🔄 최근 1시간 실제 움직임 vs 예측:")
    print("-" * 40)

    # 1시간 전 데이터
    ohlcv_1h = exchange.fetch_ohlcv('BTC/USDT', '1h', limit=2)
    price_1h_ago = ohlcv_1h[-2][4]  # 1시간 전 종가
    actual_move = ((current_price - price_1h_ago) / price_1h_ago) * 100

    actual_direction = "UP" if actual_move > 0 else "DOWN"
    predicted_direction = predictions['1h'][0]

    print(f"실제 움직임: {actual_direction} ({actual_move:+.2f}%)")
    print(f"1h 모델 예측: {predicted_direction}")
    print(f"예측 정확도: {'✅ 맞음' if actual_direction == predicted_direction else '❌ 틀림'}")

    # 최근 15분 움직임
    ohlcv_15m = exchange.fetch_ohlcv('BTC/USDT', '15m', limit=5)

    print("\n📈 최근 15분봉 추세:")
    for i in range(-5, 0):
        close = ohlcv_15m[i][4]
        change = ((close - ohlcv_15m[i-1][4]) / ohlcv_15m[i-1][4]) * 100 if i > -5 else 0
        bar = "█" * int(abs(change) * 10) if change != 0 else ""
        direction = "↑" if change > 0 else "↓" if change < 0 else "→"
        print(f"  {direction} {change:+.3f}% {bar}")

    # 예측 신뢰도 통계
    print("\n📊 신뢰도 통계:")
    print("-" * 40)

    avg_confidence = sum(c for _, c in predictions.values()) / len(predictions)
    max_conf = max((c, tf) for tf, (_, c) in predictions.items())
    min_conf = min((c, tf) for tf, (_, c) in predictions.items())

    print(f"평균 신뢰도: {avg_confidence:.1f}%")
    print(f"최고 신뢰도: {max_conf[0]:.1f}% ({max_conf[1]})")
    print(f"최저 신뢰도: {min_conf[0]:.1f}% ({min_conf[1]})")

    # 리스크 경고
    print("\n⚠️ 리스크 경고:")
    if avg_confidence < 60:
        print("  - 낮은 평균 신뢰도: 신중한 거래 필요")
    if up_count == down_count:
        print("  - 혼재된 신호: 관망 추천")
    if any(c > 90 for _, c in predictions.values()):
        high_conf = [(tf, c) for tf, (_, c) in predictions.items() if c > 90]
        for tf, c in high_conf:
            print(f"  - {tf} 과신 경고: {c:.1f}% (과적합 가능성)")

    return predictions

if __name__ == "__main__":
    test_balanced_models()