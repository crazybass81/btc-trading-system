# 🏆 최고 성능 모델 요약

**최종 업데이트: 2024-12-10 (v2.0.1)**

## 🚀 현재 운영 모델

| 타임프레임 | 전략 | 모델 타입 | 정확도 | 파일명 | 상태 |
|-----------|------|-----------|--------|--------|------|
| **30분** | Breakout | NeuralNet | **80.5%** | `breakout_30m_neuralnet_model.pkl` | ✅ 운영중 |
| **4시간** | Trend Following | NeuralNet | **77.8%** | `trend_following_4h_neuralnet_model.pkl` | ✅ 운영중 |
| **15분** | Trend Following | GradientBoost | **75.7%** | `trend_following_15m_gradientboost_model.pkl` | ✅ 운영중 |
| **1시간** | Trend Following | GradientBoost | **67.9%** | `trend_following_1h_gradientboost_model.pkl` | ✅ 운영중 |

## 📊 전체 테스트 결과 요약

### 선택된 모델들 (고신뢰도 60%+ 기준)

#### 15분봉 (15m)
| 전략 | 최고 모델 | 정확도 | 고신뢰도 정확도 | 신뢰 비율 |
|------|-----------|--------|----------------|-----------|
| **Trend Following** | GradientBoost | 71.0% | **75.7%** | 90.5% |
| **Volume Based** | XGBoost | 72.5% | **75.9%** | 70.5% |
| **Volatility** | GradientBoost | 72.0% | **71.9%** | 80.0% |
| **Pattern Recognition** | GradientBoost | 68.5% | **70.8%** | 68.5% |
| **Breakout** | GradientBoost | 66.0% | **68.6%** | 70.0% |
| **Sentiment** | GradientBoost | 68.0% | **67.1%** | 77.5% |
| **Mean Reversion** | GradientBoost | 61.0% | **66.7%** | 75.0% |
| Momentum | GradientBoost | 57.5% | 60.1% | 69.0% |

#### 30분봉 (30m)
| 전략 | 최고 모델 | 정확도 | 고신뢰도 정확도 | 신뢰 비율 |
|------|-----------|--------|----------------|-----------|
| **Breakout** | NeuralNet | 76.5% | **80.5%** | 82.0% |
| **Trend Following** | GradientBoost | 76.0% | **80.1%** | 75.5% |
| **Volume Based** | GradientBoost | 68.0% | **70.2%** | 70.5% |
| **Volatility** | GradientBoost | 63.0% | **67.6%** | 74.0% |
| **Mean Reversion** | GradientBoost | 63.0% | **65.6%** | 65.5% |
| **Pattern Recognition** | GradientBoost | 57.5% | **62.9%** | 66.0% |
| **Sentiment** | GradientBoost | 58.0% | **60.9%** | 66.5% |
| Momentum | GradientBoost | 51.5% | 53.6% | 56.0% |

#### 1시간봉 (1h)
| 전략 | 최고 모델 | 정확도 | 고신뢰도 정확도 | 신뢰 비율 |
|------|-----------|--------|----------------|-----------|
| **Trend Following** | GradientBoost | 66.5% | **67.9%** | 84.0% |
| **Volume Based** | GradientBoost | 62.5% | **67.6%** | 74.0% |
| **Breakout** | GradientBoost | 63.0% | **64.8%** | 81.0% |
| **Volatility** | GradientBoost | 61.5% | **63.9%** | 58.5% |
| **Mean Reversion** | GradientBoost | 56.5% | **60.5%** | 62.0% |
| Pattern Recognition | GradientBoost | 54.5% | 53.6% | 56.0% |
| Sentiment | GradientBoost | 57.0% | 58.8% | 58.0% |
| Momentum | GradientBoost | 56.5% | 58.9% | 47.5% |

#### 4시간봉 (4h)
| 전략 | 최고 모델 | 정확도 | 고신뢰도 정확도 | 신뢰 비율 |
|------|-----------|--------|----------------|-----------|
| **Trend Following** | NeuralNet | 73.0% | **77.8%** | 56.5% |
| **Breakout** | GradientBoost | 66.5% | **69.0%** | 54.0% |
| **Volume Based** | GradientBoost | 62.0% | **67.8%** | 51.5% |
| **Volatility** | NeuralNet | 66.0% | **67.1%** | 59.0% |
| **Mean Reversion** | GradientBoost | 64.5% | **62.6%** | 83.0% |
| Pattern Recognition | GradientBoost | 60.5% | 59.4% | 57.5% |
| Sentiment | GradientBoost | 62.5% | 59.3% | 66.0% |
| Momentum | GradientBoost | 57.5% | 59.6% | 76.0% |

## 🎯 최종 선별 모델

### 🥇 최고 성능 모델 (사용 권장)
1. **30m Breakout (NeuralNet)**: 80.5% 고신뢰도 정확도
2. **30m Trend Following (GradientBoost)**: 80.1% 고신뢰도 정확도
3. **4h Trend Following (NeuralNet)**: 77.8% 고신뢰도 정확도
4. **15m Volume Based (XGBoost)**: 75.9% 고신뢰도 정확도
5. **15m Trend Following (GradientBoost)**: 75.7% 고신뢰도 정확도

### 📈 전략별 최고 성능
- **Trend Following**: 모든 타임프레임에서 우수 (67.9% ~ 80.1%)
- **Breakout**: 30분/4시간에서 특히 우수 (69.0% ~ 80.5%)
- **Volume Based**: 15분/30분에서 우수 (70.2% ~ 75.9%)
- **Volatility**: 15분에서 가장 우수 (71.9%)

## 📁 파일 구조

### 사용 모델 (models/ 폴더)
```
models/
├── trend_following_15m_gradientboost_model.pkl
├── trend_following_15m_gradientboost_scaler.pkl
├── trend_following_30m_gradientboost_model.pkl
├── trend_following_30m_gradientboost_scaler.pkl
├── trend_following_1h_gradientboost_model.pkl
├── trend_following_1h_gradientboost_scaler.pkl
├── trend_following_4h_neuralnet_model.pkl
├── trend_following_4h_neuralnet_scaler.pkl
├── breakout_30m_neuralnet_model.pkl
├── breakout_30m_neuralnet_scaler.pkl
├── volume_based_15m_xgboost_model.pkl (pending)
└── volume_based_15m_xgboost_scaler.pkl (pending)
```

### 백업 모델 (../models/ 폴더)
성능이 60% 미만인 모델들은 ../models/ 폴더로 이동됨

## 💡 사용 방법

```python
# 최고 성능 모델 로드 예시
import joblib

# 30분 Breakout 모델 (최고 성능)
model_30m_breakout = joblib.load('models/breakout_30m_neuralnet_model.pkl')
scaler_30m_breakout = joblib.load('models/breakout_30m_neuralnet_scaler.pkl')

# 15분 Trend Following 모델
model_15m_trend = joblib.load('models/trend_following_15m_gradientboost_model.pkl')
scaler_15m_trend = joblib.load('models/trend_following_15m_gradientboost_scaler.pkl')
```

## 🔑 핵심 인사이트

1. **Trend Following 전략이 가장 안정적**: 모든 타임프레임에서 60%+ 성능
2. **30분봉이 최고 성능**: Breakout과 Trend Following에서 80%+ 달성
3. **GradientBoost가 가장 일관성 있음**: 대부분의 전략에서 최고 성능
4. **고신뢰도 거래가 핵심**: 신뢰도 70%+ 신호만 사용 시 정확도 크게 향상

## 📊 실전 거래 권장사항

### 메인 전략
- **30분 Breakout + Trend Following 조합**
- 두 모델이 같은 방향 예측 시에만 진입
- 신뢰도 70% 이상 신호만 사용

### 보조 전략
- **15분 Volume Based**: 단기 모멘텀 포착
- **4시간 Trend Following**: 중장기 방향성 확인

### 리스크 관리
- 포지션 크기: 신뢰도에 비례하여 조절
- 손절선: 2-3% (타임프레임에 따라 조정)
- 수익 실현: 부분 청산 전략 사용