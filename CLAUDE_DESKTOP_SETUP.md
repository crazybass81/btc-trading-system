# 🔌 Claude Desktop 연결 가이드
## BTC Trading System MCP Server

---

## 📋 설치 단계

### 1. 필요 패키지 설치
```bash
# MCP 서버 패키지 설치
pip install -r btc_trading_system/requirements_mcp.txt
```

### 2. Claude Desktop 설정

#### Windows
1. Claude Desktop 설정 파일 위치:
   ```
   %APPDATA%\Claude\claude_desktop_config.json
   ```

2. 설정 추가:
   ```json
   {
     "mcpServers": {
       "btc-trading": {
         "command": "python",
         "args": ["C:\\path\\to\\btc_trading_system\\mcp_server.py"],
         "env": {
           "PYTHONPATH": "C:\\path\\to\\btc_trading_system"
         }
       }
     }
   }
   ```

#### Mac
1. Claude Desktop 설정 파일 위치:
   ```
   ~/Library/Application Support/Claude/claude_desktop_config.json
   ```

2. 설정 추가:
   ```json
   {
     "mcpServers": {
       "btc-trading": {
         "command": "python3",
         "args": ["/Users/username/btc_trading_system/mcp_server.py"],
         "env": {
           "PYTHONPATH": "/Users/username/btc_trading_system"
         }
       }
     }
   }
   ```

#### Linux/WSL
1. Claude Desktop 설정 파일 위치:
   ```
   ~/.config/Claude/claude_desktop_config.json
   ```

2. 제공된 설정 사용:
   ```bash
   # 설정 파일 복사
   cp btc_trading_system/claude_desktop_config.json ~/.config/Claude/
   ```

---

## 🚀 사용 방법

### Claude Desktop에서 사용 가능한 명령어

1. **거래 신호 받기**
   ```
   "BTC 거래 신호를 알려줘"
   "지금 롱/숏 포지션을 열어도 될까?"
   ```

2. **시장 상태 확인**
   ```
   "현재 BTC 시장 상태는 어때?"
   "RSI와 지지/저항선을 확인해줘"
   ```

3. **거래 조건 체크**
   ```
   "지금 거래 조건이 충족됐나?"
   "거래 체크리스트를 확인해줘"
   ```

4. **포지션 크기 계산**
   ```
   "$10,000로 적절한 포지션 크기를 계산해줘"
   "리스크 관리 설정을 알려줘"
   ```

5. **모델 정보**
   ```
   "ML 모델의 성능은 어떻게 돼?"
   "거래 규칙을 설명해줘"
   ```

---

## 📊 제공되는 기능

### 1. `get_trading_signal()`
- 실시간 ML 예측 신호
- 신뢰도 및 거래 권장사항
- 포지션 진입/청산 가격

### 2. `get_market_status()`
- 현재 BTC 가격
- RSI 지표
- 지지/저항선
- 시장 상태 (과매수/과매도)

### 3. `check_trade_conditions()`
- 거래 체크리스트
- 신호 강도 평가
- 리스크 레벨

### 4. `get_position_size(capital)`
- 자본 대비 적정 포지션
- 손절/익절 금액
- 리스크/리워드 비율

### 5. `get_model_info()`
- 모델 성능 지표
- 거래 규칙
- 기대 수익률

---

## 🔍 테스트

### MCP 서버 직접 테스트
```bash
# 서버 실행
python btc_trading_system/mcp_server.py

# 다른 터미널에서 테스트
curl -X POST http://localhost:8000/tools/get_trading_signal
```

### Claude Desktop 연결 확인
1. Claude Desktop 재시작
2. 새 대화 시작
3. "BTC 거래 신호를 확인해줘" 입력
4. MCP 서버가 응답하는지 확인

---

## ⚠️ 주의사항

1. **API 키 필요 없음**: Binance 공개 API 사용
2. **실시간 데이터**: 15분 봉 기준
3. **모델 정확도**: 80.4% (백테스트 기준)
4. **리스크 관리**: 항상 손절선 설정

---

## 🛠️ 문제 해결

### 연결 안 될 때
1. Python 경로 확인
2. 모델 파일 존재 확인 (`models/main_15m_model.pkl`)
3. Claude Desktop 재시작

### 오류 메시지
- "모델 로드 실패": 모델 파일 경로 확인
- "API 오류": 인터넷 연결 확인
- "신호 생성 실패": Binance API 상태 확인

---

## 📱 예시 대화

```
You: BTC 거래 신호를 알려줘

Claude (with MCP): 현재 BTC 거래 신호입니다:
- 현재가: $92,010.57
- 신호: LONG
- 신뢰도: 75.5%
- 권장: ✅ 강한 신호 - 거래 가능

포지션 제안:
- 진입: $92,010.57
- 손절: $90,170.36 (-2%)
- 익절: $94,770.89 (+3%)
- 리스크/리워드: 1:1.5

예상 정확도: 92.9% (고신뢰도 신호 기준)
```

---

## 📈 성능

- **15분 모델 정확도**: 80.4%
- **고신뢰도(70%+) 정확도**: 92.9%
- **평균 수익**: +3% (성공 시)
- **평균 손실**: -2% (실패 시)
- **기대값**: +2.59% per trade

---

*개발: 2025-12-10*
*버전: 1.0*