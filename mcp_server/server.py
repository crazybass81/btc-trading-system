#!/usr/bin/env python3
"""
MCP Server for BTC Direction Prediction
LLM이 연결하여 각 시간봉별 UP/DOWN 예측을 받을 수 있는 서버
"""

from flask import Flask, jsonify, request
from flask_cors import CORS
from btc_predictor import BTCPredictor
import os
import sys
from datetime import datetime

app = Flask(__name__)
CORS(app)

# 예측기 초기화
predictor = BTCPredictor()

@app.route('/')
def home():
    """서버 상태 확인"""
    return jsonify({
        'service': 'BTC Direction Prediction MCP Server',
        'version': '1.0',
        'status': 'online',
        'models_loaded': len(predictor.models),
        'timestamp': datetime.now().isoformat()
    })

@app.route('/predict/<timeframe>/<direction>')
def predict(timeframe, direction):
    """특정 시간봉과 방향에 대한 예측

    Parameters:
    - timeframe: 15m, 30m, 1h, 4h
    - direction: up, down

    Example:
    GET /predict/1h/up
    """
    result = predictor.predict(timeframe, direction)
    return jsonify(result)

@app.route('/predict/all')
def predict_all():
    """모든 모델의 예측 반환"""
    predictions = predictor.get_all_predictions()
    return jsonify(predictions)

@app.route('/consensus')
def consensus():
    """모든 모델의 합의 예측"""
    result = predictor.get_consensus()
    return jsonify(result)

@app.route('/models')
def models():
    """사용 가능한 모델 정보"""
    info = predictor.get_model_info()
    return jsonify(info)

@app.route('/analyze')
def analyze():
    """종합 분석 (모든 정보 포함)"""
    all_predictions = predictor.get_all_predictions()
    consensus_result = predictor.get_consensus()
    model_info = predictor.get_model_info()

    # 시간대별 정리
    by_timeframe = {}
    for key, pred in all_predictions.items():
        if 'error' not in pred:
            tf = pred['timeframe']
            if tf not in by_timeframe:
                by_timeframe[tf] = {'UP': None, 'DOWN': None}
            by_timeframe[tf][pred['direction']] = pred

    return jsonify({
        'current_price': all_predictions.get(list(all_predictions.keys())[0], {}).get('current_price'),
        'consensus': consensus_result,
        'by_timeframe': by_timeframe,
        'model_performance': model_info,
        'timestamp': datetime.now().isoformat()
    })

# MCP 도구 정의 (LLM이 사용할 수 있는 함수들)
MCP_TOOLS = {
    'btc_predict': {
        'description': 'Get BTC direction prediction for specific timeframe and direction',
        'parameters': {
            'timeframe': {
                'type': 'string',
                'enum': ['15m', '30m', '1h', '4h'],
                'description': 'Time interval for prediction'
            },
            'direction': {
                'type': 'string',
                'enum': ['up', 'down'],
                'description': 'Direction to predict (up or down)'
            }
        },
        'endpoint': '/predict/{timeframe}/{direction}'
    },
    'btc_consensus': {
        'description': 'Get consensus prediction from all models',
        'parameters': {},
        'endpoint': '/consensus'
    },
    'btc_analyze': {
        'description': 'Get comprehensive BTC market analysis',
        'parameters': {},
        'endpoint': '/analyze'
    },
    'btc_models': {
        'description': 'Get information about available prediction models',
        'parameters': {},
        'endpoint': '/models'
    }
}

@app.route('/mcp/tools')
def mcp_tools():
    """MCP 도구 정의 반환 (LLM 통합용)"""
    return jsonify(MCP_TOOLS)

@app.route('/mcp/execute', methods=['POST'])
def mcp_execute():
    """MCP 도구 실행 엔드포인트"""
    data = request.json
    tool = data.get('tool')
    params = data.get('parameters', {})

    if tool == 'btc_predict':
        result = predictor.predict(
            params.get('timeframe'),
            params.get('direction')
        )
    elif tool == 'btc_consensus':
        result = predictor.get_consensus()
    elif tool == 'btc_analyze':
        all_predictions = predictor.get_all_predictions()
        consensus_result = predictor.get_consensus()
        model_info = predictor.get_model_info()

        by_timeframe = {}
        for key, pred in all_predictions.items():
            if 'error' not in pred:
                tf = pred['timeframe']
                if tf not in by_timeframe:
                    by_timeframe[tf] = {'UP': None, 'DOWN': None}
                by_timeframe[tf][pred['direction']] = pred

        result = {
            'consensus': consensus_result,
            'by_timeframe': by_timeframe,
            'model_performance': model_info,
            'timestamp': datetime.now().isoformat()
        }
    elif tool == 'btc_models':
        result = predictor.get_model_info()
    else:
        result = {'error': f'Unknown tool: {tool}'}

    return jsonify(result)

if __name__ == '__main__':
    port = int(os.environ.get('MCP_PORT', 5000))
    print(f"""
============================================================
🚀 BTC Direction Prediction MCP Server
============================================================
📊 Models loaded: {len(predictor.models)}
🌐 Server running on: http://localhost:{port}
📖 API Documentation:
   - GET /predict/<timeframe>/<direction>
   - GET /consensus
   - GET /analyze
   - GET /models
   - GET /mcp/tools (for LLM integration)
   - POST /mcp/execute (for LLM tool execution)
============================================================
    """)
    app.run(host='0.0.0.0', port=port, debug=False)