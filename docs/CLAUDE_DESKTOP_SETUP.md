# 🤖 Claude Desktop MCP Server 설정 가이드

## 📋 사전 요구사항

- Claude Desktop 최신 버전
- Python 3.9 이상
- BTC Trading System 설치 완료

---

## 🔧 설정 방법

### 1. MCP 서버 설정 파일 수정

Claude Desktop 설정 파일을 엽니다:

**macOS/Linux:**
```bash
nano ~/Library/Application\ Support/Claude/claude_desktop_config.json
```

**Windows:**
```
%APPDATA%\Claude\claude_desktop_config.json
```

### 2. BTC Trading MCP 서버 추가

다음 내용을 설정 파일에 추가:

```json
{
  "mcpServers": {
    "btc-trading": {
      "command": "python",
      "args": ["/path/to/btc_trading_system/mcp_server.py"],
      "env": {
        "PYTHONPATH": "/path/to/btc_trading_system"
      }
    }
  }
}
```

**경로 수정 필요**: `/path/to/btc_trading_system`을 실제 시스템 경로로 변경

### 3. Claude Desktop 재시작

설정 적용을 위해 Claude Desktop을 완전히 종료 후 재시작

---

## 🎯 사용 가능한 명령어

### 1. 특정 타임프레임 신호 조회
```
btc_get_signal_by_timeframe("15m")  # 15분봉 신호
btc_get_signal_by_timeframe("30m")  # 30분봉 신호
btc_get_signal_by_timeframe("4h")   # 4시간봉 신호
btc_get_signal_by_timeframe("1d")   # 1일봉 신호
```

### 2. 모든 타임프레임 분석
```
btc_get_all_timeframes()
```
**결과**: 4개 타임프레임 종합 분석 및 시장 합의도

### 3. 타임프레임 비교
```
btc_compare_timeframes()
```
**결과**: 타임프레임 간 정렬도 분석 및 거래 전략

### 4. 시장 상태 확인
```
btc_get_market_status()
```
**결과**: 현재 가격, 변화율, 거래량, 기술적 지표

### 5. 모델 정보 조회
```
btc_get_model_info()
```
**결과**: 각 모델의 정확도, 특징, 사용 가이드

---

## 💡 사용 예시

### 거래 결정 워크플로우

1. **전체 시장 분석**
```
"BTC 전체 타임프레임 분석해줘"
→ btc_get_all_timeframes() 실행
```

2. **특정 타임프레임 심화 분석**
```
"15분봉 신호 자세히 알려줘"
→ btc_get_signal_by_timeframe("15m") 실행
```

3. **타임프레임 정렬 확인**
```
"타임프레임 간 신호가 일치하나요?"
→ btc_compare_timeframes() 실행
```

4. **거래 결정**
- 정렬도 높음 (75% 이상) → 진입 고려
- 신뢰도 70% 이상 → 포지션 크기 결정
- 혼합 신호 → 관망

---

## ⚠️ 주의사항

1. **API 키 불필요**: Binance 공개 API 사용
2. **실시간 데이터**: 항상 최신 시장 데이터 기반
3. **리스크 관리**: 모든 신호에 손절선 설정 필수
4. **모델 한계**: ML 예측도 100%가 아님을 인지

---

## 🐛 문제 해결

### MCP 서버가 보이지 않음
1. 설정 파일 경로 확인
2. Python 경로 확인
3. Claude Desktop 완전 재시작

### 명령어 실행 오류
1. Python 환경 확인: `python --version`
2. 필요 패키지 설치: `pip install -r requirements.txt`
3. 모델 파일 존재 확인: `ls models/`

### 데이터 수신 실패
1. 인터넷 연결 확인
2. Binance API 상태 확인
3. 방화벽 설정 확인

---

## 📚 추가 정보

- **메인 문서**: [README.md](../README.md)
- **거래 가이드**: [GUIDE.md](GUIDE.md)
- **멀티 타임프레임**: [MULTI_TIMEFRAME_INTEGRATION.md](../claudedocs/MULTI_TIMEFRAME_INTEGRATION.md)

---

*최종 업데이트: 2024-12-10*
*MCP 서버 버전: 1.0 (멀티 타임프레임 지원)*