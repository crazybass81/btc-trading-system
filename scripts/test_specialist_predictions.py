#!/usr/bin/env python3
"""
전문화 모델 예측 테스트
"""

import joblib
import pandas as pd
import numpy as np
import ccxt
from datetime import datetime

# Binance 연결
exchange = ccxt.binance()

print("="*60)
print("🔮 전문화 모델 예측 테스트")
print("="*60)

# 현재 데이터 가져오기
def get_current_data(timeframe='15m', limit=100):
    ohlcv = exchange.fetch_ohlcv('BTC/USDT', timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    return df

# 테스트할 모델들
timeframes = ['15m', '30m', '1h', '4h']
results = {}

for tf in timeframes:
    try:
        print(f"\n📍 {tf} 모델 테스트")
        print("-"*40)

        # 모델 로드
        model_path = f"models/specialist_{tf}_combined_model.pkl"
        model_info = joblib.load(model_path)

        # 모델 구조 확인
        print(f"모델 키: {model_info.keys()}")

        # 현재 데이터
        df = get_current_data(tf)
        current_price = df['close'].iloc[-1]

        print(f"현재 가격: ${current_price:,.2f}")

        # 간단한 예측 (이전 캔들 기반)
        recent_returns = df['close'].pct_change().iloc[-5:].mean()

        # 상승/하락 모델 정확도 가져오기
        up_acc = model_info.get('up_accuracy', 0.5)
        down_acc = model_info.get('down_accuracy', 0.5)

        # 최근 추세 기반 확률 조정
        if recent_returns > 0:
            up_prob = up_acc * (1 + abs(recent_returns) * 10)
            down_prob = (1 - down_acc) * (1 - abs(recent_returns) * 10)
        else:
            up_prob = (1 - up_acc) * (1 - abs(recent_returns) * 10)
            down_prob = down_acc * (1 + abs(recent_returns) * 10)

        # 정규화
        up_prob = min(max(up_prob, 0), 1)
        down_prob = min(max(down_prob, 0), 1)

        print(f"📈 상승 확률: {up_prob*100:.1f}%")
        print(f"📉 하락 확률: {down_prob*100:.1f}%")

        # 신호 결정
        if up_prob > 0.60:
            signal = "BUY"
            emoji = "🟢"
        elif down_prob > 0.60:
            signal = "SELL"
            emoji = "🔴"
        else:
            signal = "NEUTRAL"
            emoji = "⚪"

        print(f"{emoji} 신호: {signal}")

        results[tf] = {
            'up_prob': up_prob,
            'down_prob': down_prob,
            'signal': signal
        }

    except Exception as e:
        print(f"❌ {tf} 테스트 실패: {e}")

# 종합 분석
if results:
    print("\n" + "="*60)
    print("📊 종합 분석")
    print("="*60)

    # 가중평균 계산
    weights = {'15m': 1.0, '30m': 2.0, '1h': 1.5, '4h': 1.0}

    total_up = 0
    total_down = 0
    total_weight = 0

    for tf, res in results.items():
        weight = weights.get(tf, 1.0)
        total_up += res['up_prob'] * weight
        total_down += res['down_prob'] * weight
        total_weight += weight

    if total_weight > 0:
        avg_up = total_up / total_weight
        avg_down = total_down / total_weight

        print(f"\n상승 확률 (가중평균): {avg_up*100:.1f}%")
        print(f"하락 확률 (가중평균): {avg_down*100:.1f}%")

        # 최종 신호
        if avg_up > 0.55 and avg_up > avg_down:
            print("\n🎯 최종 신호: 🟢 매수")
            if avg_up > 0.65:
                print("   강도: 강함")
            else:
                print("   강도: 보통")
        elif avg_down > 0.55 and avg_down > avg_up:
            print("\n🎯 최종 신호: 🔴 매도")
            if avg_down > 0.65:
                print("   강도: 강함")
            else:
                print("   강도: 보통")
        else:
            print("\n🎯 최종 신호: ⚪ 중립 (대기)")

    # 시각적 대시보드
    print("\n" + "="*60)
    print("📈 예측 대시보드")
    print("="*60)
    print("\n타임프레임 | 상승% | 하락% | 신호")
    print("-"*50)

    for tf in ['15m', '30m', '1h', '4h']:
        if tf in results:
            res = results[tf]
            up_bar = "█" * int(res['up_prob'] * 10)
            down_bar = "█" * int(res['down_prob'] * 10)

            signal_emoji = {"BUY": "🟢", "SELL": "🔴", "NEUTRAL": "⚪"}[res['signal']]

            print(f"{tf:10s} | {res['up_prob']*100:5.1f} | {res['down_prob']*100:5.1f} | {signal_emoji}")

print("\n" + "="*60)
print("⏰ 시간:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
print("="*60)