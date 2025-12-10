#!/usr/bin/env python3
"""
각 모델의 롱/숏 신호 빈도 분석
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.main import BTCTradingSystem
import time
import pandas as pd
from datetime import datetime
from collections import defaultdict

def analyze_signal_frequency(num_iterations=20, delay_seconds=5):
    """
    여러 번 실행하여 신호 빈도 분석
    """
    system = BTCTradingSystem()

    # 결과 저장
    results = defaultdict(lambda: {"LONG": 0, "SHORT": 0, "NEUTRAL": 0, "total": 0})
    confidence_by_signal = defaultdict(list)

    print("=" * 60)
    print("📊 모델별 신호 빈도 분석 시작")
    print(f"테스트 횟수: {num_iterations}회")
    print(f"간격: {delay_seconds}초")
    print("=" * 60)

    for i in range(num_iterations):
        print(f"\n[{i+1}/{num_iterations}] 분석 중...")

        # 각 타임프레임 분석
        for timeframe in ['15m', '30m', '4h', '1d']:
            try:
                # get_ml_prediction은 튜플을 반환 (signal, confidence)
                result = system.get_ml_prediction(timeframe)

                if result and isinstance(result, tuple) and len(result) == 2:
                    signal, confidence = result

                    if signal and signal != 'NO_MODEL':
                        # 신호 카운트
                        results[timeframe][signal] += 1
                        results[timeframe]["total"] += 1

                        # 신뢰도 기록
                        confidence_by_signal[f"{timeframe}_{signal}"].append(confidence)

                        print(f"  {timeframe}: {signal} ({confidence:.1f}%)")
                    else:
                        print(f"  {timeframe}: NO_MODEL")
                else:
                    print(f"  {timeframe}: ERROR (invalid result)")

            except Exception as e:
                print(f"  {timeframe}: 오류 - {str(e)}")

        # 다음 반복 전 대기
        if i < num_iterations - 1:
            time.sleep(delay_seconds)

    # 결과 분석
    print("\n" + "=" * 60)
    print("📈 분석 결과")
    print("=" * 60)

    summary_data = []

    for timeframe in ['15m', '30m', '4h', '1d']:
        if results[timeframe]["total"] > 0:
            total = results[timeframe]["total"]
            long_pct = (results[timeframe]["LONG"] / total) * 100
            short_pct = (results[timeframe]["SHORT"] / total) * 100
            neutral_pct = (results[timeframe]["NEUTRAL"] / total) * 100

            print(f"\n📊 {timeframe} 모델:")
            print(f"  총 신호: {total}개")
            print(f"  ├─ LONG:    {results[timeframe]['LONG']:3d}회 ({long_pct:5.1f}%)")
            print(f"  ├─ SHORT:   {results[timeframe]['SHORT']:3d}회 ({short_pct:5.1f}%)")
            print(f"  └─ NEUTRAL: {results[timeframe]['NEUTRAL']:3d}회 ({neutral_pct:5.1f}%)")

            # 평균 신뢰도
            print(f"\n  평균 신뢰도:")
            for signal_type in ['LONG', 'SHORT', 'NEUTRAL']:
                key = f"{timeframe}_{signal_type}"
                if confidence_by_signal[key]:
                    avg_conf = sum(confidence_by_signal[key]) / len(confidence_by_signal[key])
                    print(f"  ├─ {signal_type:7s}: {avg_conf:5.1f}%")

            # 요약 데이터 저장
            summary_data.append({
                'Timeframe': timeframe,
                'LONG%': long_pct,
                'SHORT%': short_pct,
                'NEUTRAL%': neutral_pct,
                'Total': total
            })

    # 테이블 형식으로 출력
    print("\n" + "=" * 60)
    print("📊 종합 요약")
    print("=" * 60)

    df = pd.DataFrame(summary_data)
    print("\n신호 분포 비율:")
    print(df.to_string(index=False))

    # 신호 일치도 분석
    print("\n" + "=" * 60)
    print("🎯 신호 경향성 분석")
    print("=" * 60)

    for timeframe in ['15m', '30m', '4h', '1d']:
        if results[timeframe]["total"] > 0:
            total = results[timeframe]["total"]
            directional = results[timeframe]["LONG"] + results[timeframe]["SHORT"]
            directional_pct = (directional / total) * 100

            if directional > 0:
                long_ratio = results[timeframe]["LONG"] / directional
                bias = "LONG" if long_ratio > 0.5 else "SHORT"
                bias_strength = max(long_ratio, 1-long_ratio) * 100

                print(f"\n{timeframe}:")
                print(f"  방향성 신호 비율: {directional_pct:.1f}%")
                print(f"  경향성: {bias} ({bias_strength:.1f}% 우세)")

    # 현재 시장 상태와 비교
    print("\n" + "=" * 60)
    print("💡 해석")
    print("=" * 60)
    print("""
    1. NEUTRAL이 많은 경우:
       - 시장이 횡보/불확실한 상황
       - 모델이 명확한 방향성을 찾지 못함

    2. LONG/SHORT 편향이 강한 경우:
       - 해당 타임프레임에서 명확한 추세 존재
       - 높은 신뢰도로 거래 기회

    3. 타임프레임별 차이:
       - 단기(15m, 30m): 변동성 높아 신호 자주 변경
       - 장기(4h, 1d): 안정적인 추세 신호
    """)

    return results

if __name__ == "__main__":
    # 20번 테스트, 5초 간격
    results = analyze_signal_frequency(num_iterations=20, delay_seconds=5)

    print("\n✅ 분석 완료!")
    print("=" * 60)