#!/usr/bin/env python3
"""
BTC Direction Prediction MCP Server
Provides MCP protocol-compliant access to BTC prediction models

This server exposes trained ML models for BTC price direction prediction
through the Model Context Protocol (MCP), enabling LLM integration.
"""

import os
import sys
import json
import logging
from typing import Dict, Optional, List, Any, Literal
from datetime import datetime
from enum import Enum
from fastmcp import FastMCP
from pydantic import BaseModel, Field, validator

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the predictor
from btc_predictor import BTCPredictor

# Configure logging to stderr (MCP uses stdout for protocol)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stderr  # Important: MCP uses stdout for JSON-RPC
)
logger = logging.getLogger(__name__)

# Initialize MCP server
mcp = FastMCP(
    name="btc-predictor",
    version="1.0.0"
)

# Global predictor instance
predictor = None


class TimeFrame(str, Enum):
    """Supported timeframes for BTC prediction"""
    FIFTEEN_MIN = "15m"
    THIRTY_MIN = "30m"
    ONE_HOUR = "1h"
    FOUR_HOUR = "4h"


class Direction(str, Enum):
    """Prediction directions"""
    UP = "up"
    DOWN = "down"


class ResponseFormat(str, Enum):
    """Response format options"""
    JSON = "json"
    MARKDOWN = "markdown"


class GetPredictionRequest(BaseModel):
    """Request model for getting a prediction"""
    timeframe: TimeFrame = Field(
        ...,
        description="Timeframe for prediction (15m, 30m, 1h, 4h)"
    )
    direction: Direction = Field(
        ...,
        description="Direction to predict (up or down)"
    )
    format: ResponseFormat = Field(
        default=ResponseFormat.JSON,
        description="Response format (json or markdown)"
    )

    @validator('timeframe')
    def validate_timeframe(cls, v):
        if v not in ['15m', '30m', '1h', '4h']:
            raise ValueError(f"Invalid timeframe: {v}")
        return v


class GetConsensusRequest(BaseModel):
    """Request model for consensus prediction"""
    format: ResponseFormat = Field(
        default=ResponseFormat.JSON,
        description="Response format (json or markdown)"
    )


class AnalyzeMarketRequest(BaseModel):
    """Request model for comprehensive market analysis"""
    include_individual: bool = Field(
        default=True,
        description="Include individual model predictions"
    )
    format: ResponseFormat = Field(
        default=ResponseFormat.MARKDOWN,
        description="Response format (json or markdown)"
    )


class GetModelInfoRequest(BaseModel):
    """Request model for model information"""
    format: ResponseFormat = Field(
        default=ResponseFormat.JSON,
        description="Response format (json or markdown)"
    )


def format_prediction_markdown(prediction: Dict) -> str:
    """Format prediction result as markdown"""
    if 'error' in prediction:
        return f"❌ **Error**: {prediction['error']}"

    # Choose emoji based on signal
    signal = prediction.get('signal', prediction.get('prediction', 'NO_SIGNAL'))
    if signal == 'UP':
        direction_emoji = "📈"
        signal_text = "**UP Signal**"
    elif signal == 'DOWN':
        direction_emoji = "📉"
        signal_text = "**DOWN Signal**"
    else:
        direction_emoji = "⏸️"
        signal_text = "**No Signal**"

    confidence_level = "High" if prediction['confidence'] > 0.75 else "Medium" if prediction['confidence'] > 0.6 else "Low"

    # 실질적 신뢰도 계산 및 표시
    real_conf = prediction.get('real_confidence', prediction['confidence'] * prediction['model_accuracy'] / 100)
    real_conf_level = "High" if real_conf > 0.6 else "Medium" if real_conf > 0.45 else "Low"

    return f"""## {direction_emoji} BTC {prediction['timeframe']} {prediction['direction']} Model

**Signal**: {signal_text}
**Signal Strength**: {prediction.get('signal_strength', prediction['confidence']):.1%}
**Prediction Confidence**: {prediction['confidence']:.1%} ({confidence_level})
**Model Accuracy**: {prediction['model_accuracy']}%
**Real Confidence**: {real_conf:.1%} ({real_conf_level}) ⭐
**Current Price**: ${prediction['current_price']:,.2f}
**Timestamp**: {prediction['timestamp']}

### Interpretation
- The {prediction['direction']} model is {"generating a" if signal != "NO_SIGNAL" else "not generating any"} {signal_text.lower()}
- Signal strength: {prediction.get('signal_strength', prediction['confidence']):.1%}
- **Real confidence** (prediction × accuracy): {real_conf:.1%}
- This means the actual probability of this prediction being correct is approximately **{real_conf:.1%}**"""


def format_consensus_markdown(consensus: Dict) -> str:
    """Format consensus result as markdown"""
    if 'error' in consensus:
        return f"❌ **Error**: {consensus['error']}"

    # Choose emoji based on consensus
    if consensus['consensus'] == 'UP':
        emoji = "🚀"
        action = "Bullish"
    elif consensus['consensus'] == 'DOWN':
        emoji = "🔻"
        action = "Bearish"
    elif consensus['consensus'] == 'NO_SIGNAL':
        emoji = "⏸️"
        action = "No Active Signals"
    else:
        emoji = "⚖️"
        action = "Neutral"

    return f"""## {emoji} BTC Market Consensus: {action}

**Consensus Direction**: {consensus['consensus']}
**Prediction Confidence**: {consensus['confidence']:.1%}
**Real Confidence**: {consensus.get('real_confidence', 0):.1%} ⭐
**UP Score**: {consensus.get('up_score', consensus.get('up_probability', 0)):.1%} (Real: {consensus.get('real_up_score', 0):.1%})
**DOWN Score**: {consensus.get('down_score', consensus.get('down_probability', 0)):.1%} (Real: {consensus.get('real_down_score', 0):.1%})
**Active Signals**: {', '.join(consensus.get('active_signals', [])) if consensus.get('active_signals') else 'None'}
**Total Models**: {consensus['total_models']}
**Timestamp**: {consensus['timestamp']}

### Signal Strength
{get_signal_strength_analysis(consensus)}"""


def get_signal_strength_analysis(consensus: Dict) -> str:
    """Analyze and describe signal strength based on real confidence"""
    # Use real confidence if available, otherwise fall back to regular confidence
    real_confidence = consensus.get('real_confidence', consensus['confidence'])
    confidence = consensus['confidence']

    # 실질적 신뢰도 기반 분석
    if real_confidence > 0.6:
        return f"✅ **Strong Signal** - Real confidence {real_confidence:.1%} (>60% actual probability)"
    elif real_confidence > 0.5:
        return f"🟡 **Moderate Signal** - Real confidence {real_confidence:.1%} (50-60% actual probability)"
    elif real_confidence > 0.4:
        return f"🟠 **Weak Signal** - Real confidence {real_confidence:.1%} (40-50% actual probability)"
    else:
        return f"⚪ **No Clear Signal** - Real confidence {real_confidence:.1%} (<40% actual probability)"


def format_market_analysis_markdown(all_predictions: Dict, consensus: Dict) -> str:
    """Format comprehensive market analysis as markdown"""
    analysis = f"""# 🎯 BTC Market Analysis Report

## 📊 Overall Market Consensus
{format_consensus_markdown(consensus)}

## 📈 Individual Model Predictions
"""

    # Group by timeframe
    timeframes = {'15m': [], '30m': [], '1h': [], '4h': []}

    for key, pred in all_predictions.items():
        if 'error' not in pred:
            tf = pred['timeframe']
            if tf in timeframes:
                timeframes[tf].append(pred)

    # Sort each timeframe by accuracy
    for tf in timeframes:
        if timeframes[tf]:
            timeframes[tf].sort(key=lambda x: x['model_accuracy'], reverse=True)

    # Format each timeframe
    for tf_name, preds in timeframes.items():
        if preds:
            analysis += f"\n### {tf_name} Timeframe\n"
            for pred in preds:
                emoji = "📈" if pred['prediction'] == 'UP' else "📉"
                analysis += f"- {emoji} **{pred['direction']}**: {pred['prediction']} ({pred['confidence']:.1%} confidence, {pred['model_accuracy']}% accuracy)\n"

    # Add interpretation guide
    analysis += """
## 📖 How to Interpret These Signals

### Strong Signals (Recommended for Trading)
- ✅ Both 1h UP/DOWN models agree on direction
- ✅ Consensus confidence > 75%
- ✅ Trading during optimal times (21:00, 01:00, 00:00 UTC)

### Moderate Signals
- 🟡 30m + 1h models agree
- 🟡 Consensus confidence 65-75%

### Weak Signals (Use Caution)
- 🟠 Only 15m models signaling
- 🟠 Consensus confidence < 60%
- 🟠 Models disagree across timeframes
"""

    return analysis


def format_model_info_markdown(info: Dict) -> str:
    """Format model information as markdown"""
    md = f"""# 🤖 BTC Prediction Models Overview

## 📊 Summary Statistics
- **Total Models**: {info['total']}
- **Average Accuracy**: {info['average_accuracy']:.1f}%

## 🎯 Individual Models (Sorted by Accuracy)
"""

    for model in info['models']:
        emoji = "🥇" if model['accuracy'] > 75 else "🥈" if model['accuracy'] > 70 else "🥉"
        md += f"\n### {emoji} {model['name']}\n"
        md += f"- **Timeframe**: {model['timeframe']}\n"
        md += f"- **Direction**: {model['direction']}\n"
        md += f"- **Accuracy**: {model['accuracy']}%\n"

    md += """
## 📈 Model Performance Notes
- All models trained on 45+ days of historical data
- Validated through backtesting on 90-day periods
- Deep Ensemble models use 14 sub-models for stability
- Models achieve 60%+ accuracy requirement
"""

    return md


@mcp.tool(
    description="Get BTC price direction prediction for a specific timeframe and direction"
)
async def btc_get_prediction(request: GetPredictionRequest) -> str:
    """
    Get prediction from a specific BTC direction model.

    Returns prediction with confidence level and model accuracy.
    """
    global predictor

    if predictor is None:
        predictor = BTCPredictor()

    try:
        # Get prediction
        result = predictor.predict(request.timeframe, request.direction)

        # Format response based on requested format
        if request.format == ResponseFormat.MARKDOWN:
            return format_prediction_markdown(result)
        else:
            return json.dumps(result, indent=2)

    except Exception as e:
        logger.error(f"Error getting prediction: {e}")
        error_result = {"error": str(e)}
        if request.format == ResponseFormat.MARKDOWN:
            return f"❌ **Error**: {str(e)}"
        else:
            return json.dumps(error_result, indent=2)


@mcp.tool(
    description="Get consensus prediction from all BTC models with weighted voting"
)
async def btc_get_consensus(request: GetConsensusRequest) -> str:
    """
    Get weighted consensus from all available BTC prediction models.

    Combines predictions from all timeframes with accuracy-based weighting.
    """
    global predictor

    if predictor is None:
        predictor = BTCPredictor()

    try:
        # Get consensus
        result = predictor.get_consensus()

        # Format response
        if request.format == ResponseFormat.MARKDOWN:
            return format_consensus_markdown(result)
        else:
            return json.dumps(result, indent=2)

    except Exception as e:
        logger.error(f"Error getting consensus: {e}")
        error_result = {"error": str(e)}
        if request.format == ResponseFormat.MARKDOWN:
            return f"❌ **Error**: {str(e)}"
        else:
            return json.dumps(error_result, indent=2)


@mcp.tool(
    description="Get comprehensive BTC market analysis with all model predictions"
)
async def btc_analyze_market(request: AnalyzeMarketRequest) -> str:
    """
    Perform comprehensive market analysis using all available models.

    Provides individual predictions, consensus, and interpretation guide.
    """
    global predictor

    if predictor is None:
        predictor = BTCPredictor()

    try:
        # Get all predictions and consensus
        all_predictions = predictor.get_all_predictions()
        consensus = predictor.get_consensus()

        if request.format == ResponseFormat.MARKDOWN:
            return format_market_analysis_markdown(all_predictions, consensus)
        else:
            # JSON format
            result = {
                "consensus": consensus,
                "individual_predictions": all_predictions if request.include_individual else None,
                "timestamp": datetime.now().isoformat()
            }
            return json.dumps(result, indent=2)

    except Exception as e:
        logger.error(f"Error analyzing market: {e}")
        error_result = {"error": str(e)}
        if request.format == ResponseFormat.MARKDOWN:
            return f"❌ **Error during market analysis**: {str(e)}"
        else:
            return json.dumps(error_result, indent=2)


@mcp.tool(
    description="Get information about available BTC prediction models and their performance"
)
async def btc_get_model_info(request: GetModelInfoRequest) -> str:
    """
    Get detailed information about all available prediction models.

    Includes model names, timeframes, directions, and historical accuracy.
    """
    global predictor

    if predictor is None:
        predictor = BTCPredictor()

    try:
        # Get model info
        result = predictor.get_model_info()

        # Format response
        if request.format == ResponseFormat.MARKDOWN:
            return format_model_info_markdown(result)
        else:
            return json.dumps(result, indent=2)

    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        error_result = {"error": str(e)}
        if request.format == ResponseFormat.MARKDOWN:
            return f"❌ **Error**: {str(e)}"
        else:
            return json.dumps(error_result, indent=2)


@mcp.resource(
    uri="btc://models/overview",
    name="BTC Models Overview"
)
async def get_models_overview() -> str:
    """Resource providing overview of all BTC prediction models"""
    global predictor

    if predictor is None:
        predictor = BTCPredictor()

    info = predictor.get_model_info()
    return json.dumps(info, indent=2)


@mcp.resource(
    uri="btc://predictions/latest",
    name="Latest BTC Predictions"
)
async def get_latest_predictions() -> str:
    """Resource providing latest predictions from all models"""
    global predictor

    if predictor is None:
        predictor = BTCPredictor()

    all_predictions = predictor.get_all_predictions()
    consensus = predictor.get_consensus()

    result = {
        "consensus": consensus,
        "predictions": all_predictions,
        "timestamp": datetime.now().isoformat()
    }

    return json.dumps(result, indent=2)


# Server initialization
def initialize_server():
    """Initialize the MCP server and predictor"""
    global predictor

    logger.info("Initializing BTC Prediction MCP Server...")

    try:
        # Initialize predictor
        predictor = BTCPredictor()
        logger.info(f"✅ Loaded {len(predictor.models)} models successfully")

        # Log available models
        for key in predictor.models.keys():
            config = predictor.models[key]['config']
            logger.info(f"  - {config['name']}: {config['accuracy']}% accuracy")

        return True

    except Exception as e:
        logger.error(f"Failed to initialize server: {e}")
        return False


if __name__ == "__main__":
    # Initialize server
    if initialize_server():
        logger.info("🚀 BTC Prediction MCP Server ready!")
        logger.info("Server: btc-predictor v1.0.0")
        logger.info("Tools available:")
        logger.info("  - btc_get_prediction: Get specific model prediction")
        logger.info("  - btc_get_consensus: Get weighted consensus")
        logger.info("  - btc_analyze_market: Comprehensive market analysis")
        logger.info("  - btc_get_model_info: Model performance info")

        # Run FastMCP server (stdio mode for Claude Desktop)
        mcp.run()
    else:
        logger.error("Failed to start server")
        sys.exit(1)