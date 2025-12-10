#!/usr/bin/env python3
"""
훈련 데이터의 실제 라벨 분포 확인
"""

import pandas as pd
import numpy as np
import ccxt
from datetime import datetime, timedelta

def create_labels(df, threshold=0.002):
    """레이블 생성 (훈련 스크립트와 동일)"""
    future_return = df['close'].shift(-1) / df['close'] - 1
    labels = pd.Series(1, index=df.index)  # 기본값 NEUTRAL (1)
    labels[future_return > threshold] = 2  # LONG (2)
    labels[future_return < -threshold] = 0  # SHORT (0)
    return labels

def analyze_label_distribution():
    """다양한 기간의 라벨 분포 분석"""
    exchange = ccxt.binance()

    print("=" * 60)
    print("📊 BTC 가격 변화 및 라벨 분포 분석")
    print("=" * 60)

    timeframes = ['15m', '30m', '4h', '1d']
    limits = [2000, 1000, 500, 365]

    for tf, limit in zip(timeframes, limits):
        print(f"\n🕐 {tf} 타임프레임 (최근 {limit}개 캔들)")
        print("-" * 40)

        try:
            # 데이터 가져오기
            ohlcv = exchange.fetch_ohlcv('BTC/USDT', timeframe=tf, limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

            # 가격 변화율 계산
            df['price_change'] = df['close'].pct_change()

            # 라벨 생성
            labels = create_labels(df, threshold=0.002)  # 0.2% 임계값

            # 분포 계산
            label_counts = labels.value_counts().sort_index()
            total = len(labels.dropna())

            # 결과 출력
            label_map = {0: 'SHORT', 1: 'NEUTRAL', 2: 'LONG'}

            for label_val in [0, 1, 2]:
                count = label_counts.get(label_val, 0)
                pct = (count / total) * 100 if total > 0 else 0
                print(f"  {label_map[label_val]:8s}: {count:4d}개 ({pct:5.1f}%)")

            # 가격 변화 통계
            price_changes = df['price_change'].dropna() * 100

            print(f"\n  가격 변화 통계:")
            print(f"  평균: {price_changes.mean():+.3f}%")
            print(f"  표준편차: {price_changes.std():.3f}%")
            print(f"  최대 상승: {price_changes.max():+.3f}%")
            print(f"  최대 하락: {price_changes.min():+.3f}%")

            # 임계값 초과 비율
            above_threshold = (price_changes.abs() > 0.2).sum()
            above_pct = (above_threshold / len(price_changes)) * 100
            print(f"  ±0.2% 초과 변동: {above_threshold}개 ({above_pct:.1f}%)")

            # 최근 24시간 분포
            recent_labels = labels.iloc[-96:] if tf == '15m' else labels.iloc[-24:]
            recent_counts = recent_labels.value_counts().sort_index()

            print(f"\n  최근 24시간:")
            for label_val in [0, 1, 2]:
                count = recent_counts.get(label_val, 0)
                pct = (count / len(recent_labels)) * 100 if len(recent_labels) > 0 else 0
                print(f"  {label_map[label_val]:8s}: {count:3d}개 ({pct:5.1f}%)")

        except Exception as e:
            print(f"  오류: {e}")

    # 다양한 임계값 테스트
    print("\n" + "=" * 60)
    print("🔍 임계값별 분포 변화 (15분봉 기준)")
    print("=" * 60)

    ohlcv = exchange.fetch_ohlcv('BTC/USDT', timeframe='15m', limit=2000)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

    thresholds = [0.001, 0.002, 0.003, 0.005, 0.01]  # 0.1%, 0.2%, 0.3%, 0.5%, 1%

    for threshold in thresholds:
        labels = create_labels(df, threshold=threshold)
        label_counts = labels.value_counts().sort_index()
        total = len(labels.dropna())

        print(f"\n임계값 {threshold*100:.1f}%:")

        for label_val in [0, 1, 2]:
            count = label_counts.get(label_val, 0)
            pct = (count / total) * 100 if total > 0 else 0
            label_name = {0: 'SHORT', 1: 'NEUTRAL', 2: 'LONG'}[label_val]
            print(f"  {label_name:8s}: {pct:5.1f}%", end="")

            # 시각적 바 차트
            bar_length = int(pct / 2)
            print(f"  {'█' * bar_length}")

    # 제안 사항
    print("\n" + "=" * 60)
    print("💡 분석 결과 및 제안")
    print("=" * 60)
    print("""
    1. 현재 상황:
       - 0.2% 임계값은 15분봉에서 너무 높음
       - 대부분의 가격 변화가 ±0.2% 이내 → NEUTRAL 편향

    2. 문제점:
       - NEUTRAL이 너무 많으면 거래 신호가 거의 없음
       - 모델이 안전한 NEUTRAL만 예측하도록 학습됨

    3. 해결 방법:
       a) 임계값 조정: 0.1% 또는 0.15%로 낮춤
       b) 동적 임계값: 변동성에 따라 조정
       c) 클래스 가중치: 소수 클래스에 더 높은 가중치
       d) 다른 라벨링 방법: 이동평균 기반, 지지/저항 기반

    4. 권장사항:
       - 15분: 0.1% 임계값
       - 30분: 0.15% 임계값
       - 4시간: 0.3% 임계값
       - 1일: 0.5% 임계값
    """)

if __name__ == "__main__":
    analyze_label_distribution()