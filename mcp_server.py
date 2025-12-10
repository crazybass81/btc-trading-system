#!/usr/bin/env python3
"""
BTC Trading System MCP Server
Claude Desktop과 연결하기 위한 MCP 서버
"""

import sys
import os
from typing import Any, Dict, Optional
from datetime import datetime
from pydantic import BaseModel, Field

# 시스템 경로 추가
sys.path.append(os.path.join(os.path.dirname(__file__), 'core'))

# FastMCP 설치 필요: pip install fastmcp
from fastmcp import FastMCP

# 메인 시스템 import
from main import BTCTradingSystem

# ===== Pydantic Models for Type Safety =====

class TradingSignalResponse(BaseModel):
    """Trading signal response model"""
    timestamp: str
    current_price: Optional[float]
    signal: str
    confidence: str
    action: str
    recommendation: str
    expected_accuracy: str
    technical_indicators: Dict[str, Any]
    position_advice: Optional[Dict[str, Any]]
    model_info: Dict[str, str]

class MarketStatusResponse(BaseModel):
    """Market status response model"""
    timestamp: str
    current_price: str
    rsi: str
    support: str
    resistance: str
    market_condition: str
    hint: str
    price_range: Dict[str, Any]

class TradeConditionsResponse(BaseModel):
    """Trade conditions response model"""
    can_trade: bool
    checklist: Dict[str, str]
    recommendation: str
    risk_level: str
    signal: str
    confidence: str
    current_price: Optional[str]

class PositionSizeResponse(BaseModel):
    """Position size calculation response model"""
    total_capital: str
    recommended_position: Dict[str, str]
    risk_management: Dict[str, str]
    current_btc_price: str
    notes: list[str]

class ModelInfoResponse(BaseModel):
    """Model information response model"""
    model_name: str = Field(description="Name of the ML model")
    version: str = Field(description="Model version")
    training_date: str = Field(description="Date when model was trained")
    performance: Dict[str, str] = Field(description="Performance metrics")
    features: list[str] = Field(description="Features used in model")
    trading_rules: Dict[str, str] = Field(description="Trading rules")
    expected_performance: Dict[str, str] = Field(description="Expected performance")
    disclaimer: str = Field(description="Risk disclaimer")

# ===== MCP Server Setup =====

# MCP 서버 초기화 (naming convention: underscore for package name)
mcp = FastMCP("btc_trading_mcp", dependencies=["ccxt", "pandas", "numpy", "scikit-learn"])
mcp.description = "BTC Trading System - ML-based trading signal generator with 80.4% accuracy"

# 거래 시스템 인스턴스 (singleton pattern)
_trading_system: Optional[BTCTradingSystem] = None

def get_system() -> BTCTradingSystem:
    """거래 시스템 인스턴스 가져오기 (singleton)"""
    global _trading_system
    if _trading_system is None:
        _trading_system = BTCTradingSystem()
    return _trading_system

# ===== Tool Implementations =====

@mcp.tool()
def btc_get_trading_signal() -> Dict[str, Any]:
    """
    Generate BTC trading signals using ML models.

    Returns a comprehensive trading signal including:
    - Current price and market status
    - Signal direction (LONG/SHORT/NEUTRAL)
    - Confidence level and trading recommendation
    - Position advice with entry/exit points
    - Model performance information
    """
    system = get_system()

    try:
        # ML 예측
        signal, confidence = system.get_ml_prediction('15m')

        # 기술적 지표
        tech = system.get_technical_indicators()

        # 거래 결정 로직
        if confidence >= 70:
            action = "TRADE"
            recommendation = "✅ Strong signal - Trade recommended"
            expected_accuracy = "92.9%"
        elif confidence >= 65:
            action = "CAUTION"
            recommendation = "⚠️ Moderate signal - Use caution"
            expected_accuracy = "75%"
        else:
            action = "NO_TRADE"
            recommendation = "❌ Weak signal - Do not trade"
            expected_accuracy = "Low"

        # 포지션 제안 생성
        position_advice = None
        if action == "TRADE" and tech:
            if signal == "LONG":
                position_advice = {
                    "type": "LONG",
                    "entry": tech['current_price'],
                    "stop_loss": round(tech['current_price'] * 0.98, 2),
                    "take_profit": round(tech['current_price'] * 1.03, 2),
                    "risk_reward": "1:1.5"
                }
            elif signal == "SHORT":
                position_advice = {
                    "type": "SHORT",
                    "entry": tech['current_price'],
                    "stop_loss": round(tech['current_price'] * 1.02, 2),
                    "take_profit": round(tech['current_price'] * 0.97, 2),
                    "risk_reward": "1:1.5"
                }

        response = TradingSignalResponse(
            timestamp=datetime.now().isoformat(),
            current_price=tech['current_price'] if tech else None,
            signal=signal,
            confidence=f"{confidence:.1f}%",
            action=action,
            recommendation=recommendation,
            expected_accuracy=expected_accuracy,
            technical_indicators={
                "rsi": round(tech['rsi'], 1) if tech else None,
                "support": round(tech['support'], 2) if tech else None,
                "resistance": round(tech['resistance'], 2) if tech else None
            },
            position_advice=position_advice,
            model_info={
                "model": "15-minute ML model",
                "backtest_accuracy": "80.4%",
                "high_confidence_accuracy": "92.9%"
            }
        )

        return response.model_dump()

    except Exception as e:
        return {
            "error": str(e),
            "message": "Failed to generate trading signal. Please try again later."
        }

@mcp.tool()
def btc_get_market_status() -> Dict[str, Any]:
    """
    Get current BTC market status with technical indicators.

    Returns:
    - Current BTC price
    - RSI indicator value
    - Support and resistance levels
    - Market condition assessment
    """
    system = get_system()

    try:
        tech = system.get_technical_indicators()

        if tech:
            # RSI 기반 시장 상태 판단
            if tech['rsi'] < 30:
                market_condition = "Oversold"
                market_hint = "Potential bounce"
            elif tech['rsi'] > 70:
                market_condition = "Overbought"
                market_hint = "Potential correction"
            else:
                market_condition = "Neutral"
                market_hint = "Waiting for direction"

            response = MarketStatusResponse(
                timestamp=datetime.now().isoformat(),
                current_price=f"${tech['current_price']:,.2f}",
                rsi=f"{tech['rsi']:.1f}",
                support=f"${tech['support']:,.2f}",
                resistance=f"${tech['resistance']:,.2f}",
                market_condition=market_condition,
                hint=market_hint,
                price_range={
                    "low": tech['support'],
                    "high": tech['resistance'],
                    "range_percent": f"{((tech['resistance'] - tech['support']) / tech['current_price'] * 100):.1f}%"
                }
            )

            return response.model_dump()
        else:
            return {"error": "Unable to fetch market data"}

    except Exception as e:
        return {"error": str(e)}

@mcp.tool()
def btc_check_trade_conditions() -> Dict[str, Any]:
    """
    Check current trading conditions against entry requirements.

    Returns:
    - Trade readiness assessment
    - Checklist of conditions
    - Risk level evaluation
    - Trading recommendations
    """
    system = get_system()

    try:
        # 신호 확인
        signal, confidence = system.get_ml_prediction('15m')
        tech = system.get_technical_indicators()

        # 체크리스트 생성
        checklist = {
            "signal_clarity": "✅" if signal in ["LONG", "SHORT"] else "❌",
            "confidence_above_70": "✅" if confidence >= 70 else "❌",
            "rsi_in_range": "✅" if tech and 30 <= tech['rsi'] <= 70 else "⚠️",
            "risk_management": "✅ Stop loss -2%, Take profit +3%",
            "position_sizing": "✅ Max 5% of capital",
            "time_limit": "✅ Close within 4 hours"
        }

        # 거래 가능 여부 판단
        can_trade = (
            signal in ["LONG", "SHORT"] and
            confidence >= 70
        )

        # 권장사항 생성
        if can_trade:
            recommendation = "Conditions met - Can enter position"
            risk_level = "Low"
        elif confidence >= 65:
            recommendation = "Partial conditions met - Small position or wait"
            risk_level = "Medium"
        else:
            recommendation = "Conditions not met - Do not trade"
            risk_level = "High"

        response = TradeConditionsResponse(
            can_trade=can_trade,
            checklist=checklist,
            recommendation=recommendation,
            risk_level=risk_level,
            signal=signal,
            confidence=f"{confidence:.1f}%",
            current_price=f"${tech['current_price']:,.2f}" if tech else None
        )

        return response.model_dump()

    except Exception as e:
        return {"error": str(e)}

@mcp.tool()
def btc_calculate_position_size(capital: float = 10000) -> Dict[str, Any]:
    """
    Calculate appropriate position size based on capital.

    Args:
        capital: Total trading capital in USD (default: $10,000)

    Returns:
    - Recommended position size in USD and BTC
    - Risk management calculations
    - Maximum loss and profit expectations
    """
    system = get_system()

    try:
        tech = system.get_technical_indicators()

        if tech:
            # 포지션 크기 계산 (자본의 5%)
            position_size = capital * 0.05

            # BTC 수량 계산
            btc_amount = position_size / tech['current_price']

            # 리스크 계산
            stop_loss_amount = position_size * 0.02  # -2%
            take_profit_amount = position_size * 0.03  # +3%

            response = PositionSizeResponse(
                total_capital=f"${capital:,.2f}",
                recommended_position={
                    "usd": f"${position_size:,.2f}",
                    "btc": f"{btc_amount:.6f} BTC",
                    "percent_of_capital": "5%"
                },
                risk_management={
                    "max_loss": f"${stop_loss_amount:.2f}",
                    "max_loss_percent": "2%",
                    "expected_profit": f"${take_profit_amount:.2f}",
                    "expected_profit_percent": "3%",
                    "risk_reward_ratio": "1:1.5"
                },
                current_btc_price=f"${tech['current_price']:,.2f}",
                notes=[
                    "Position size is 5% of capital (recommended)",
                    "Always set stop loss at -2%",
                    "Close position within 4 hours"
                ]
            )

            return response.model_dump()
        else:
            return {"error": "Unable to fetch market data"}

    except Exception as e:
        return {"error": str(e)}

@mcp.tool()
def btc_get_model_info() -> Dict[str, Any]:
    """
    Get ML model information and performance metrics.

    Returns:
    - Model specifications and version
    - Performance metrics from backtesting
    - Trading rules and parameters
    - Expected performance statistics
    """
    response = ModelInfoResponse(
        model_name="BTC 15-minute Prediction Model",
        version="1.0",
        training_date="2024-12-10",
        performance={
            "overall_accuracy": "80.4%",
            "high_confidence_accuracy": "92.9%",
            "confidence_threshold": "70%",
            "backtest_period": "14 days",
            "total_predictions": "1344"
        },
        features=[
            "Price change rate (1, 3, 5, 10 candles)",
            "RSI (7, 14, 21)",
            "MACD",
            "Bollinger Bands",
            "Volume indicators"
        ],
        trading_rules={
            "entry": "Confidence ≥ 70%",
            "stop_loss": "-2%",
            "take_profit": "+3%",
            "max_holding": "4 hours",
            "position_size": "5% of capital"
        },
        expected_performance={
            "win_rate": "~93% (high confidence signals)",
            "avg_profit": "+3%",
            "avg_loss": "-2%",
            "expected_value": "+2.59% per trade"
        },
        disclaimer="Past performance does not guarantee future results. Risk management is essential."
    )

    return response.model_dump()

# ===== Server Initialization =====

def main():
    """Main entry point for MCP server"""
    print("🚀 BTC Trading System MCP Server")
    print("📊 Model: 15-minute ML model (80.4% accuracy)")
    print("🔗 Ready for Claude Desktop connection")
    print("-" * 50)

    # Run the MCP server
    mcp.run()

if __name__ == "__main__":
    main()