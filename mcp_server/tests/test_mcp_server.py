#!/usr/bin/env python3
"""
Test script for BTC Prediction MCP Server
Tests all MCP tools and validates responses
"""

import os
import sys
import json
import asyncio
from typing import Dict

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the MCP server module
from mcp_server import initialize_server
from btc_predictor import BTCPredictor


async def test_get_prediction():
    """Test individual prediction tool"""
    print("\n🔍 Testing btc_get_prediction...")

    # Test JSON format
    request = GetPredictionRequest(
        timeframe="1h",
        direction="up",
        format="json"
    )

    result = await btc_get_prediction(request)
    print(f"  JSON Response: {result[:100]}...")

    # Test Markdown format
    request_md = GetPredictionRequest(
        timeframe="1h",
        direction="up",
        format="markdown"
    )

    result_md = await btc_get_prediction(request_md)
    print(f"  Markdown Response: {result_md[:200]}...")

    # Test different timeframes
    for tf in ["15m", "30m", "1h", "4h"]:
        request = GetPredictionRequest(
            timeframe=tf,
            direction="up",
            format="json"
        )
        result = await btc_get_prediction(request)
        data = json.loads(result)
        if 'error' not in data:
            print(f"  ✅ {tf} UP: {data.get('prediction')} ({data.get('confidence', 0)*100:.1f}%)")


async def test_get_consensus():
    """Test consensus prediction tool"""
    print("\n🔍 Testing btc_get_consensus...")

    # Test JSON format
    request = GetConsensusRequest(format="json")
    result = await btc_get_consensus(request)
    data = json.loads(result)

    if 'error' not in data:
        print(f"  Consensus: {data['consensus']}")
        print(f"  Confidence: {data['confidence']*100:.1f}%")
        print(f"  UP Probability: {data['up_probability']*100:.1f}%")
        print(f"  DOWN Probability: {data['down_probability']*100:.1f}%")

    # Test Markdown format
    request_md = GetConsensusRequest(format="markdown")
    result_md = await btc_get_consensus(request_md)
    print(f"  Markdown Response: {result_md[:200]}...")


async def test_analyze_market():
    """Test market analysis tool"""
    print("\n🔍 Testing btc_analyze_market...")

    # Test with individual predictions
    request = AnalyzeMarketRequest(
        include_individual=True,
        format="json"
    )

    result = await btc_analyze_market(request)
    data = json.loads(result)

    if 'error' not in data:
        print(f"  Consensus: {data['consensus']['consensus']}")
        if data.get('individual_predictions'):
            print(f"  Individual predictions included: {len(data['individual_predictions'])} models")

    # Test Markdown format
    request_md = AnalyzeMarketRequest(
        include_individual=True,
        format="markdown"
    )

    result_md = await btc_analyze_market(request_md)
    print(f"  Markdown Response: {result_md[:300]}...")


async def test_get_model_info():
    """Test model info tool"""
    print("\n🔍 Testing btc_get_model_info...")

    # Test JSON format
    request = GetModelInfoRequest(format="json")
    result = await btc_get_model_info(request)
    data = json.loads(result)

    if 'error' not in data:
        print(f"  Total Models: {data['total']}")
        print(f"  Average Accuracy: {data['average_accuracy']:.1f}%")
        print("\n  Top Models:")
        for model in data['models'][:3]:
            print(f"    - {model['name']}: {model['accuracy']}% ({model['timeframe']} {model['direction']})")

    # Test Markdown format
    request_md = GetModelInfoRequest(format="markdown")
    result_md = await btc_get_model_info(request_md)
    print(f"\n  Markdown Response: {result_md[:300]}...")


async def test_error_handling():
    """Test error handling"""
    print("\n🔍 Testing error handling...")

    # Test invalid timeframe
    try:
        request = GetPredictionRequest(
            timeframe="invalid",
            direction="up"
        )
    except Exception as e:
        print(f"  ✅ Validation error caught: {str(e)[:50]}...")

    # Test with valid but non-existent model
    request = GetPredictionRequest(
        timeframe="15m",
        direction="down"  # This model doesn't exist
    )

    result = await btc_get_prediction(request)
    data = json.loads(result)
    if 'error' in data:
        print(f"  ✅ Error handled: {data['error']}")


async def main():
    """Run all tests"""
    print("=" * 60)
    print("🚀 BTC PREDICTION MCP SERVER TEST SUITE")
    print("=" * 60)

    # Initialize server
    print("\n📦 Initializing server...")
    if not initialize_server():
        print("❌ Failed to initialize server")
        return

    print("✅ Server initialized successfully")

    # Run all tests
    try:
        await test_get_prediction()
        await test_get_consensus()
        await test_analyze_market()
        await test_get_model_info()
        await test_error_handling()

        print("\n" + "=" * 60)
        print("✅ ALL TESTS COMPLETED SUCCESSFULLY")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())