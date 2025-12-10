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
mcp.description = "BTC Multi-Timeframe Trading System - ML models for 15m (80.4%), 30m (72.1%), 4h (78.6%), 1d (75.0%)"

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
            if signal == "UP":
                position_advice = {
                    "type": "LONG",
                    "entry": tech['current_price'],
                    "stop_loss": round(tech['current_price'] * 0.98, 2),
                    "take_profit": round(tech['current_price'] * 1.03, 2),
                    "risk_reward": "1:1.5"
                }
            elif signal == "DOWN":
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
            "signal_clarity": "✅" if signal in ["UP", "DOWN"] else "❌",
            "confidence_above_70": "✅" if confidence >= 70 else "❌",
            "rsi_in_range": "✅" if tech and 30 <= tech['rsi'] <= 70 else "⚠️",
            "risk_management": "✅ Stop loss -2%, Take profit +3%",
            "position_sizing": "✅ Max 5% of capital",
            "time_limit": "✅ Close within 4 hours"
        }

        # 거래 가능 여부 판단
        can_trade = (
            signal in ["UP", "DOWN"] and
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
def btc_get_signal_by_timeframe(timeframe: str = "15m") -> Dict[str, Any]:
    """
    Get trading signal for a specific timeframe.

    Args:
        timeframe: Timeframe to analyze - "15m", "30m", "4h", or "1d"

    Returns:
    - Trading signal for the specified timeframe
    - Model accuracy and performance metrics
    - Position advice with entry/exit points
    """
    system = get_system()

    # 타임프레임 검증
    valid_timeframes = {
        '15m': ('15-minute', 75.7, '단기 트레이딩'),
        '30m': ('30-minute', 80.5, '중기 트레이딩'),
        '1h': ('1-hour', 67.9, '중장기 트레이딩'),
        '4h': ('4-hour', 77.8, '장기 추세')
    }

    if timeframe not in valid_timeframes:
        return {
            "error": f"Invalid timeframe. Choose from: {', '.join(valid_timeframes.keys())}"
        }

    try:
        # ML 예측
        signal, confidence = system.get_ml_prediction(timeframe)
        name, accuracy, description = valid_timeframes[timeframe]

        # 기술적 지표 (해당 타임프레임)
        tech = system.get_technical_indicators()

        # 거래 결정 로직
        if confidence >= 70:
            action = "TRADE"
            recommendation = f"✅ Strong {name} signal - Trade recommended"
        elif confidence >= 65:
            action = "CAUTION"
            recommendation = f"⚠️ Moderate {name} signal - Use caution"
        else:
            action = "NO_TRADE"
            recommendation = f"❌ Weak {name} signal - Do not trade"

        # 포지션 제안 (TRADE 신호일 때만)
        position_advice = None
        if action == "TRADE" and tech and signal in ["UP", "DOWN"]:
            if signal == "UP":
                position_advice = {
                    "type": "LONG",
                    "entry": tech['current_price'],
                    "stop_loss": round(tech['current_price'] * 0.98, 2),
                    "take_profit": round(tech['current_price'] * 1.03, 2),
                    "risk_reward": "1:1.5"
                }
            elif signal == "DOWN":
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
            expected_accuracy=f"{accuracy}%",
            technical_indicators={
                "rsi": round(tech['rsi'], 1) if tech else None,
                "support": round(tech['support'], 2) if tech else None,
                "resistance": round(tech['resistance'], 2) if tech else None
            },
            position_advice=position_advice,
            model_info={
                "model": f"{name} model",
                "timeframe": timeframe,
                "backtest_accuracy": f"{accuracy}%",
                "description": description
            }
        )

        return response.model_dump()

    except Exception as e:
        return {
            "error": str(e),
            "message": f"Failed to generate {timeframe} signal. Please try again."
        }

@mcp.tool()
def btc_get_all_timeframes() -> Dict[str, Any]:
    """
    Get trading signals for all available timeframes at once.

    Returns:
    - Signals for 15m, 30m, 4h, and 1d timeframes
    - Overall market consensus
    - Multi-timeframe analysis summary
    """
    system = get_system()

    timeframes = {
        '15m': ('15-minute', 75.7, '단기'),
        '30m': ('30-minute', 80.5, '중기'),
        '1h': ('1-hour', 67.9, '중장기'),
        '4h': ('4-hour', 77.8, '장기')
    }

    signals = {}
    long_count = 0
    short_count = 0
    neutral_count = 0

    try:
        for tf, (name, accuracy, desc) in timeframes.items():
            try:
                signal, confidence = system.get_ml_prediction(tf)

                signals[tf] = {
                    "name": name,
                    "signal": signal,
                    "confidence": f"{confidence:.1f}%",
                    "accuracy": f"{accuracy}%",
                    "description": desc
                }

                # 신호 카운트 (UP/DOWN 신호로 변경)
                if signal == "UP":
                    long_count += 1
                elif signal == "DOWN":
                    short_count += 1
                else:
                    neutral_count += 1

            except Exception as e:
                signals[tf] = {"error": str(e)}

        # 종합 판단
        total_signals = long_count + short_count + neutral_count
        if long_count > short_count and long_count > neutral_count:
            consensus = "BULLISH"
            consensus_strength = f"{(long_count / total_signals * 100):.0f}%"
        elif short_count > long_count and short_count > neutral_count:
            consensus = "BEARISH"
            consensus_strength = f"{(short_count / total_signals * 100):.0f}%"
        else:
            consensus = "NEUTRAL"
            consensus_strength = "Mixed signals"

        # 현재 가격
        tech = system.get_technical_indicators()

        return {
            "timestamp": datetime.now().isoformat(),
            "current_price": f"${tech['current_price']:,.2f}" if tech else None,
            "signals": signals,
            "consensus": {
                "direction": consensus,
                "strength": consensus_strength,
                "long_signals": long_count,
                "short_signals": short_count,
                "neutral_signals": neutral_count
            },
            "recommendation": (
                f"Multi-timeframe analysis shows {consensus} bias. "
                f"{long_count} LONG, {short_count} SHORT, {neutral_count} NEUTRAL signals detected."
            )
        }

    except Exception as e:
        return {
            "error": str(e),
            "message": "Failed to generate multi-timeframe analysis."
        }

@mcp.tool()
def btc_compare_timeframes() -> Dict[str, Any]:
    """
    Compare signals across all timeframes for trend confirmation.

    Returns:
    - Detailed comparison of all timeframe signals
    - Trend strength analysis
    - Trading strategy recommendation based on timeframe alignment
    """
    system = get_system()

    timeframes = {
        '15m': ('Short-term (15m)', 75.7, '스캘핑/데이트레이딩'),
        '30m': ('Mid-term (30m)', 80.5, '스윙 트레이딩'),
        '1h': ('Mid-Long (1h)', 67.9, '중장기 트레이딩'),
        '4h': ('Long-term (4h)', 77.8, '포지션 트레이딩')
    }

    comparison = []
    alignment_score = 0

    try:
        signals_list = []
        for tf, (name, accuracy, strategy) in timeframes.items():
            try:
                signal, confidence = system.get_ml_prediction(tf)
                signals_list.append(signal)

                # 신호 강도 평가
                if confidence >= 70:
                    strength = "Strong"
                elif confidence >= 65:
                    strength = "Moderate"
                else:
                    strength = "Weak"

                comparison.append({
                    "timeframe": tf,
                    "name": name,
                    "signal": signal,
                    "confidence": f"{confidence:.1f}%",
                    "strength": strength,
                    "accuracy": f"{accuracy}%",
                    "strategy": strategy
                })

            except Exception as e:
                comparison.append({
                    "timeframe": tf,
                    "error": str(e)
                })

        # 타임프레임 정렬 분석
        if len(signals_list) >= 3:
            # 모든 신호가 같은 방향이면 강한 정렬
            if len(set(signals_list)) == 1:
                alignment = "Perfect Alignment"
                alignment_score = 100
                recommendation = f"🎯 All timeframes agree on {signals_list[0]}. High confidence trade setup."
            # 대부분 같은 방향
            elif signals_list.count("UP") >= 3:
                alignment = "Strong Bullish"
                alignment_score = 75
                recommendation = "📈 Multiple timeframes show UP bias. Consider bullish position."
            elif signals_list.count("DOWN") >= 3:
                alignment = "Strong Bearish"
                alignment_score = 75
                recommendation = "📉 Multiple timeframes show DOWN bias. Consider bearish position."
            # 혼재
            else:
                alignment = "Mixed Signals"
                alignment_score = 50
                recommendation = "⚠️ Timeframes show conflicting signals. Wait for clearer setup."
        else:
            alignment = "Insufficient Data"
            alignment_score = 0
            recommendation = "Not enough signals for analysis."

        # 현재 가격
        tech = system.get_technical_indicators()

        return {
            "timestamp": datetime.now().isoformat(),
            "current_price": f"${tech['current_price']:,.2f}" if tech else None,
            "timeframe_comparison": comparison,
            "alignment_analysis": {
                "alignment": alignment,
                "score": alignment_score,
                "recommendation": recommendation
            },
            "trading_strategy": {
                "short_term": "Use 15m for precise entries",
                "mid_term": "Use 30m for swing trades",
                "long_term": "Use 4h for trend direction",
                "confirmation": "1d for overall market bias"
            }
        }

    except Exception as e:
        return {
            "error": str(e),
            "message": "Failed to compare timeframes."
        }

@mcp.tool()
def btc_get_model_info() -> Dict[str, Any]:
    """
    Get ML model information and performance metrics for all timeframes.

    Returns:
    - Model specifications for all timeframes
    - Performance metrics from backtesting
    - Trading rules and parameters
    - Timeframe-specific recommendations
    """
    system = get_system()

    # 모든 타임프레임 모델 정보
    models_info = {
        "15m": {
            "name": "15-minute Trend Following",
            "accuracy": "75.7%",
            "description": "단기 트레이딩 (스캘핑/데이트레이딩)",
            "use_case": "Quick entries/exits, scalping",
            "holding_time": "15 min - 4 hours",
            "best_for": "Day traders, scalpers"
        },
        "30m": {
            "name": "30-minute Breakout",
            "accuracy": "80.5%",
            "description": "중기 트레이딩 (스윙) - 최고 성능",
            "use_case": "Breakout trading, intraday positions",
            "holding_time": "1 - 8 hours",
            "best_for": "Swing traders"
        },
        "1h": {
            "name": "1-hour Trend Following",
            "accuracy": "67.9%",
            "description": "중장기 트레이딩",
            "use_case": "Medium-term trend following",
            "holding_time": "4 - 12 hours",
            "best_for": "Medium-term traders"
        },
        "4h": {
            "name": "4-hour Trend Following",
            "accuracy": "77.8%",
            "description": "장기 추세 (포지션 트레이딩)",
            "use_case": "Trend following, position trading",
            "holding_time": "1 - 7 days",
            "best_for": "Position traders, trend followers"
        }
    }

    # 로드된 모델 확인
    loaded_models = list(system.models.keys())

    return {
        "system_name": "BTC Multi-Timeframe Trading System",
        "version": "2.0",
        "training_date": "2024-12-10",
        "loaded_models": loaded_models,
        "models": models_info,
        "features": [
            "Price change rate (1, 3, 5, 10 candles)",
            "RSI (7, 14, 21)",
            "MACD (12, 26, 9)",
            "Bollinger Bands (10, 20)",
            "Volume indicators",
            "High-Low ratio",
            "Close position"
        ],
        "trading_rules": {
            "entry": "Confidence ≥ 70%",
            "stop_loss": "-2%",
            "take_profit": "+3%",
            "position_size": "5% of capital",
            "risk_reward": "1:1.5"
        },
        "multi_timeframe_strategy": {
            "confirmation": "Use 1d for overall market direction",
            "trend": "Use 4h for trend confirmation",
            "entry": "Use 15m/30m for precise entry timing",
            "alignment": "Best trades occur when all timeframes align"
        },
        "expected_performance": {
            "high_confidence_win_rate": "~90%+",
            "avg_profit": "+3%",
            "avg_loss": "-2%",
            "expected_value": "+2.5% per trade"
        },
        "disclaimer": "Past performance does not guarantee future results. Risk management is essential. Always use stop losses."
    }

# ===== Server Initialization =====

def main():
    """Main entry point for MCP server"""
    print("=" * 60)
    print("🚀 BTC Multi-Timeframe Trading System MCP Server")
    print("=" * 60)
    print("📊 Available Models:")
    print("  • 15분 모델: 75.7% 정확도 (단기 추세)")
    print("  • 30분 모델: 80.5% 정확도 (Breakout - 최고 성능)")
    print("  • 1시간 모델: 67.9% 정확도 (중기 추세)")
    print("  • 4시간 모델: 77.8% 정확도 (장기 추세)")
    print("-" * 60)
    print("🔧 Available Tools:")
    print("  • btc_get_trading_signal() - 15분 신호 (기본)")
    print("  • btc_get_signal_by_timeframe(tf) - 특정 타임프레임")
    print("  • btc_get_all_timeframes() - 모든 타임프레임 분석")
    print("  • btc_compare_timeframes() - 타임프레임 비교")
    print("  • btc_get_market_status() - 시장 상태")
    print("  • btc_check_trade_conditions() - 거래 조건 확인")
    print("  • btc_calculate_position_size() - 포지션 크기")
    print("  • btc_get_model_info() - 모델 정보")
    print("=" * 60)
    print("🔗 Ready for Claude Desktop connection")
    print("=" * 60)

    # Run the MCP server
    mcp.run()

if __name__ == "__main__":
    main()