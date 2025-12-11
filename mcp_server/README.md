# BTC Direction Prediction MCP Server

LLM이 연결하여 BTC 방향성 예측을 받을 수 있는 MCP (Model Context Protocol) 서버

## 🎯 Server Options

### 1. **MCP Protocol Server** (Recommended) - `mcp_server.py`
Full MCP protocol-compliant implementation using FastMCP framework with proper tool registration and Pydantic validation.

### 2. **REST API Server** (Legacy) - `server.py`
Flask-based REST API for HTTP integration.

## 🚀 빠른 시작 (MCP Protocol Server)

### 1. 설치
```bash
pip install -r requirements.txt
pip install fastmcp pydantic
```

### 2. MCP 서버 실행
```bash
# MCP Protocol Server (Recommended)
./run_mcp.sh
# 또는 직접 실행
python -m fastmcp run mcp_server.py
```

### 3. 테스트 (MCP Server)
```bash
# MCP server test
python -c "from mcp_server import btc_get_prediction; print(await btc_get_prediction({'timeframe': '1h', 'direction': 'up'}))"
```

## 🚀 빠른 시작 (REST API Server - Legacy)

### 1. 서버 실행
```bash
python server.py
# 또는 포트 지정
MCP_PORT=5001 python server.py
```

### 2. 테스트
```bash
# 서버 상태 확인
curl http://localhost:5000/

# 1시간봉 상승 예측
curl http://localhost:5000/predict/1h/up

# 합의 예측
curl http://localhost:5000/consensus

# 종합 분석
curl http://localhost:5000/analyze
```

## 📊 API 엔드포인트

### 개별 예측
```
GET /predict/<timeframe>/<direction>
```
- **timeframe**: 15m, 30m, 1h, 4h
- **direction**: up, down

**응답 예시:**
```json
{
  "timeframe": "1h",
  "direction": "UP",
  "prediction": "UP",
  "confidence": 0.796,
  "model_accuracy": 79.6,
  "current_price": 90345.74,
  "timestamp": "2025-12-11T12:00:00"
}
```

### 합의 예측
```
GET /consensus
```

**응답 예시:**
```json
{
  "consensus": "UP",
  "confidence": 0.65,
  "up_probability": 0.65,
  "down_probability": 0.35,
  "total_models": 8,
  "timestamp": "2025-12-11T12:00:00"
}
```

### 종합 분석
```
GET /analyze
```

**응답 예시:**
```json
{
  "current_price": 90345.74,
  "consensus": {
    "consensus": "UP",
    "confidence": 0.65
  },
  "by_timeframe": {
    "15m": {
      "UP": {...},
      "DOWN": null
    },
    "30m": {
      "UP": {...},
      "DOWN": {...}
    }
  },
  "model_performance": {
    "models": [...],
    "average_accuracy": 72.4
  }
}
```

### 모델 정보
```
GET /models
```

**응답 예시:**
```json
{
  "models": [
    {
      "timeframe": "1h",
      "direction": "UP",
      "accuracy": 79.6,
      "name": "deep_ensemble_1h_up"
    },
    ...
  ],
  "total": 8,
  "average_accuracy": 72.4
}
```

## 🤖 LLM 통합

### MCP 도구 정의
```
GET /mcp/tools
```

### MCP 도구 실행
```
POST /mcp/execute
Content-Type: application/json

{
  "tool": "btc_predict",
  "parameters": {
    "timeframe": "1h",
    "direction": "up"
  }
}
```

### 사용 가능한 도구
- `btc_predict`: 특정 시간봉/방향 예측
- `btc_consensus`: 합의 예측
- `btc_analyze`: 종합 분석
- `btc_models`: 모델 정보

## 📋 모델 성능

| 모델 | 시간봉 | 방향 | 정확도 |
|------|--------|------|--------|
| Deep Ensemble | 1h | UP | 79.6% |
| Deep Ensemble | 1h | DOWN | 78.7% |
| Deep Ensemble | 4h | UP | 75.9% |
| Deep Ensemble | 4h | DOWN | 74.1% |
| Deep Ensemble | 30m | UP | 72.9% |
| Deep Ensemble | 30m | DOWN | 70.4% |
| Advanced ML | 15m | UP | 65.2% |
| Deep Ensemble | 15m | UP | 62.8% |

**평균 정확도: 72.4%**

## 🔧 환경 변수

- `MCP_PORT`: 서버 포트 (기본: 5000)

## 📦 Docker 실행 (선택사항)

```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 5000
CMD ["python", "server.py"]
```

```bash
docker build -t btc-mcp-server .
docker run -p 5000:5000 btc-mcp-server
```

## 🔗 LLM 연동 예시

### Claude/GPT 연동
```python
import requests

def get_btc_prediction(timeframe, direction):
    response = requests.get(f"http://localhost:5000/predict/{timeframe}/{direction}")
    return response.json()

# 사용 예시
prediction = get_btc_prediction("1h", "up")
print(f"1시간 상승 예측: {prediction['confidence']*100:.1f}%")
```

### MCP 프로토콜 통합
```json
{
  "name": "btc-predictor",
  "version": "1.0",
  "tools": [
    {
      "name": "btc_predict",
      "description": "Get BTC direction prediction",
      "parameters": {
        "timeframe": "string",
        "direction": "string"
      }
    }
  ],
  "endpoint": "http://localhost:5000/mcp/execute"
}
```

## 📞 문의

문제가 있으면 GitHub Issues에 제보해주세요.