# 📡 Signal Separation Update

## 🎯 Overview
MCP 서버의 신호 생성 로직이 개선되어 각 모델이 올바른 신호만 생성하도록 업데이트되었습니다.

## 🔄 주요 변경 사항

### 1. 신호 분리 로직
- **UP 모델**: UP 신호 또는 NO_SIGNAL만 생성
- **DOWN 모델**: DOWN 신호 또는 NO_SIGNAL만 생성
- 각 모델이 자신의 방향과 반대되는 신호를 생성하지 않음

### 2. 새로운 응답 필드
```json
{
    "signal": "UP/DOWN/NO_SIGNAL",      // 실제 신호
    "signal_strength": 0.75,            // 신호 강도
    "confidence": 0.75,                 // 모델 신뢰도
    "direction": "UP/DOWN",             // 모델 타입
    // ... 기타 필드
}
```

### 3. 합의(Consensus) 개선
- **active_signals**: 현재 활성화된 신호 목록 표시
- **up_score/down_score**: UP/DOWN 점수 분리 계산
- **NO_SIGNAL 상태**: 모든 모델이 신호를 생성하지 않을 때

## 📊 신호 해석 가이드

### UP 모델 동작
```
UP 모델 신뢰도 > 50%
  → UP 신호 생성
  → signal_strength = confidence

UP 모델 신뢰도 ≤ 50%
  → NO_SIGNAL
  → signal_strength = 1 - confidence
```

### DOWN 모델 동작
```
DOWN 모델 신뢰도 > 50%
  → DOWN 신호 생성
  → signal_strength = confidence

DOWN 모델 신뢰도 ≤ 50%
  → NO_SIGNAL
  → signal_strength = 1 - confidence
```

## 💡 사용 예시

### MCP Tool 호출
```python
# UP 모델에서 UP 신호만 받기
result = btc_get_prediction(timeframe="1h", direction="up")
if result['signal'] == 'UP':
    print(f"1h UP 신호 발생! 강도: {result['signal_strength']:.1%}")

# DOWN 모델에서 DOWN 신호만 받기
result = btc_get_prediction(timeframe="1h", direction="down")
if result['signal'] == 'DOWN':
    print(f"1h DOWN 신호 발생! 강도: {result['signal_strength']:.1%}")
```

### 합의 분석
```python
consensus = btc_get_consensus()
print(f"활성 신호: {', '.join(consensus['active_signals'])}")
print(f"UP 점수: {consensus['up_score']:.1%}")
print(f"DOWN 점수: {consensus['down_score']:.1%}")
```

## 🔍 테스트 결과

### 신호 분리 테스트
```
✅ 1h UP 모델: UP 신호만 생성 (71.6%)
✅ 1h DOWN 모델: DOWN 신호만 생성 (86.6%)
✅ 4h UP 모델: UP 신호만 생성 (68.3%)
✅ 4h DOWN 모델: DOWN 신호만 생성 (81.5%)
✅ 30m UP 모델: UP 신호만 생성 (80.2%)
✅ 30m DOWN 모델: DOWN 신호만 생성 (63.4%)
✅ 15m UP 모델: UP 신호만 생성 (58.7%)

모든 테스트 통과 (7/7) ✅
```

## 🎯 장점

1. **명확한 신호**: 각 모델이 자신의 전문 분야만 예측
2. **혼란 방지**: UP 모델이 DOWN 신호를 생성하는 혼란 제거
3. **더 나은 해석**: 각 모델의 역할이 명확히 구분됨
4. **신호 추적**: active_signals로 현재 활성 신호 쉽게 확인

## 📝 업데이트 요약

- **날짜**: 2025-12-11
- **버전**: 1.1.0
- **상태**: ✅ 프로덕션 준비 완료
- **테스트**: 모든 모델 정상 작동 확인

---

이제 MCP 서버가 더 직관적이고 명확한 신호를 제공합니다!