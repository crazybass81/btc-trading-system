# 다음 단계 개선 제안

## 1. 즉시 적용 가능한 개선

### 1.1 4h DOWN 모델 통합
```python
# live_trading_strategy.py에 추가
{
    'name': 'deep_ensemble_4h_down',
    'path': 'models/deep_ensemble_4h_down_model.pkl',
    'accuracy': 74.1,
    'timeframe': '4h',
    'direction': 'DOWN',
    'best_hours': [8, 20],  # 분석 후 결정
}
```

### 1.2 신호 필터 강화
```python
def filter_signals(consensus):
    """노이즈 제거 필터"""
    # 1. 극단적 불일치 제거
    if short_term_up and long_term_down:
        return "HOLD"  # 추세 전환 구간

    # 2. 시간대 보정
    if in_optimal_hours:
        confidence *= 1.1  # 10% 가중치

    # 3. 변동성 조정
    if volatility > threshold:
        confidence *= 0.9  # 고변동성시 보수적
```

## 2. 중기 개선 (1-2주)

### 2.1 15m DOWN 재훈련
- 데이터 60일로 확대
- 클래스 불균형 해결 (SMOTE)
- 노이즈 필터링 강화
- 목표: 60% 이상 달성

### 2.2 실시간 성과 추적
```python
class PerformanceTracker:
    def __init__(self):
        self.predictions = []
        self.actuals = []

    def track(self, pred, actual):
        """예측 vs 실제 추적"""
        self.predictions.append(pred)
        self.actuals.append(actual)

    def report(self):
        """일일 성과 리포트"""
        accuracy = accuracy_score(self.actuals, self.predictions)
        return {
            'accuracy': accuracy,
            'signals': len(self.predictions),
            'profit': calculate_profit()
        }
```

### 2.3 동적 가중치 조정
```python
def update_weights(model_performance):
    """실시간 성과 기반 가중치 조정"""
    for model in models:
        recent_accuracy = get_recent_accuracy(model, days=7)
        if recent_accuracy < historical_accuracy * 0.9:
            model.weight *= 0.8  # 성능 하락시 가중치 감소
```

## 3. 장기 개선 (1개월+)

### 3.1 시장 체제 인식
```python
class MarketRegimeDetector:
    def detect(self):
        """불/베어, 변동성 체제 감지"""
        if volatility > 80th_percentile:
            return "HIGH_VOL"
        elif trend_strength > threshold:
            return "TRENDING"
        else:
            return "RANGING"

    def adjust_models(self, regime):
        """체제별 모델 조정"""
        if regime == "HIGH_VOL":
            use_short_term_models_only()
        elif regime == "TRENDING":
            use_long_term_models_only()
```

### 3.2 멀티 타겟 학습
```python
# 현재: 방향만 예측
# 개선: 방향 + 강도 + 지속시간
targets = {
    'direction': 'UP/DOWN',
    'strength': '1-5%',
    'duration': '1-10 candles'
}
```

### 3.3 외부 데이터 통합
- 거래소 펀딩 레이트
- 온체인 데이터 (고래 움직임)
- 소셜 센티먼트
- 매크로 지표 (DXY, 금)

## 4. 백테스트 검증 필요

### 4.1 Walk-Forward Analysis
```python
# 3개월 훈련 → 1개월 테스트 → 반복
for period in periods:
    train_on(period.train_data)
    test_on(period.test_data)
    record_performance()
```

### 4.2 몬테카를로 시뮬레이션
```python
# 1000회 랜덤 거래 시뮬레이션
for i in range(1000):
    random_trades = generate_random_trades()
    results.append(simulate_pnl(random_trades))
confidence_interval = np.percentile(results, [5, 95])
```

## 5. 리스크 관리 강화

### 5.1 켈리 공식 적용
```python
def kelly_criterion(win_prob, win_loss_ratio):
    """최적 베팅 크기 계산"""
    f = (win_prob * win_loss_ratio - (1 - win_prob)) / win_loss_ratio
    return max(0, min(f, 0.25))  # 최대 25% 제한
```

### 5.2 최대 손실 제한
```python
class RiskManager:
    def __init__(self, max_daily_loss=0.05):
        self.max_daily_loss = max_daily_loss
        self.daily_pnl = 0

    def can_trade(self):
        return self.daily_pnl > -self.max_daily_loss

    def update(self, pnl):
        self.daily_pnl += pnl
```

## 6. 실행 우선순위

1. **즉시**: 4h DOWN 모델 통합 ✅
2. **이번주**: 실시간 추적 시스템 구축
3. **다음주**: 15m DOWN 재훈련
4. **이번달**: 시장 체제 인식 추가
5. **장기**: 외부 데이터 통합