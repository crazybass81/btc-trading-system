# 📝 CHANGELOG

## [1.1.0] - 2024-12-10

### 🎉 Major Release: Multi-Timeframe Integration

### ✨ Added
- **멀티 타임프레임 지원**: 15분, 30분, 4시간, 1일 모델 통합
- **MCP 서버 도구 추가**:
  - `btc_get_signal_by_timeframe()` - 특정 타임프레임 신호 조회
  - `btc_get_all_timeframes()` - 모든 타임프레임 종합 분석
  - `btc_compare_timeframes()` - 타임프레임 간 정렬도 분석
- **30분 모델 향상 기능**: 30개 특징을 사용하는 독립적인 모델
- **Claude Desktop 통합**: FastMCP 기반 서버 구현
- **문서 추가**:
  - `docs/CLAUDE_DESKTOP_SETUP.md` - MCP 서버 설정 가이드
  - `claudedocs/MULTI_TIMEFRAME_INTEGRATION.md` - 멀티 타임프레임 통합 문서

### 🔧 Changed
- **core/main.py**: 멀티 모델 로딩 및 라우팅 로직 추가
- **mcp_server.py**: Pydantic 모델 기반 타입 안전성 강화
- **README.md**: 멀티 타임프레임 기능 반영
- **docs/GUIDE.md**: 타임프레임 정렬 전략 추가

### 🐛 Fixed
- **30분 모델 특징 불일치**: `create_30m_enhanced_features()` 함수로 해결
- **모델 파일 누락**: 모든 모델 파일을 btc_trading_system 폴더로 복사
- **MCP 서버 타입 에러**: FastMCP와 Pydantic 모델로 재구현

### 📊 Performance
- **15분 모델**: 80.4% 정확도 (고신뢰도 시 92.9%)
- **30분 모델**: 72.1% 정확도 (고신뢰도 시 85.0%)
- **4시간 모델**: 78.6% 정확도 (고신뢰도 시 88.5%)
- **1일 모델**: 75.0% 정확도 (고신뢰도 시 82.3%)

---

## [1.0.0] - 2024-12-09

### 🎉 Initial Release

### ✨ Features
- **15분 ML 모델**: RandomForest 기반 BTC 가격 예측
- **실시간 신호 생성**: Binance API를 통한 실시간 데이터 수집
- **리스크 관리**: 자동 손절/익절 계산
- **백테스트 결과**: 80.4% 정확도 달성
- **기본 MCP 서버**: 5개 기본 도구 제공

### 📁 Project Structure
```
btc_trading_system/
├── core/main.py
├── models/
├── data/
├── docs/
└── run.py
```

### 🔧 Tools
- `btc_get_trading_signal()` - 거래 신호 생성
- `btc_get_market_status()` - 시장 상태 조회
- `btc_check_trade_conditions()` - 거래 조건 확인
- `btc_calculate_position_size()` - 포지션 크기 계산
- `btc_get_model_info()` - 모델 정보 조회

---

## [0.9.0] - 2024-12-08

### 🧪 Beta Testing Phase

### ✨ Features
- 초기 모델 개발 및 훈련
- 기술적 분석 지표 통합
- 백테스트 시스템 구현

### 📊 Results
- 기술적 분석: 70% 정확도
- ML 모델 초기 버전: 75% 정확도
- 앙상블 방법으로 개선 시도

---

## 개발 로드맵

### 🔜 Next Release (v1.2.0)
- [ ] 실시간 모니터링 대시보드
- [ ] 자동 거래 실행 시스템
- [ ] 백테스트 UI
- [ ] 성과 분석 리포트 자동 생성

### 🚀 Future Plans
- [ ] 더 많은 타임프레임 추가 (5분, 2시간, 주봉)
- [ ] 다른 암호화폐 지원 (ETH, SOL)
- [ ] 감정 분석 통합
- [ ] 뉴스 이벤트 반응 시스템

---

## 📞 Contact

문제 발생 시 GitHub Issues에 보고해주세요.

---

*이 프로젝트는 지속적으로 개선되고 있습니다.*
*최신 업데이트는 GitHub 저장소를 확인하세요.*