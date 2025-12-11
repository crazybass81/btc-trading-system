#!/usr/bin/env python3
"""
Simple test for BTC Prediction MCP Server
Tests the predictor functionality directly
"""

import os
import sys
import json

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from btc_predictor import BTCPredictor


def main():
    """Run simple tests"""
    print("=" * 60)
    print("🚀 BTC PREDICTION SIMPLE TEST")
    print("=" * 60)

    # Initialize predictor
    print("\n📦 Initializing predictor...")
    predictor = BTCPredictor()

    # Test individual predictions
    print("\n🔍 Testing individual predictions:")
    for timeframe in ['15m', '30m', '1h', '4h']:
        for direction in ['up', 'down']:
            result = predictor.predict(timeframe, direction)
            if 'error' not in result:
                print(f"  ✅ {timeframe:3} {direction:4}: {result['prediction']} ({result['confidence']*100:.1f}%)")
            else:
                print(f"  ❌ {timeframe:3} {direction:4}: {result['error']}")

    # Test consensus
    print("\n🔍 Testing consensus:")
    consensus = predictor.get_consensus()
    if 'error' not in consensus:
        print(f"  Consensus: {consensus['consensus']}")
        print(f"  Confidence: {consensus['confidence']*100:.1f}%")
        print(f"  UP Probability: {consensus['up_probability']*100:.1f}%")
        print(f"  DOWN Probability: {consensus['down_probability']*100:.1f}%")

    # Test model info
    print("\n🔍 Testing model info:")
    info = predictor.get_model_info()
    print(f"  Total Models: {info['total']}")
    print(f"  Average Accuracy: {info['average_accuracy']:.1f}%")

    print("\n" + "=" * 60)
    print("✅ TEST COMPLETED")
    print("=" * 60)

    # MCP Server Info
    print("\n📡 MCP Server Info:")
    print("  The MCP server is ready for use!")
    print("  REST API: Run 'python server.py' (port 5001)")
    print("  MCP Protocol: Run './run_mcp.sh' (FastMCP)")
    print("\n  Available tools:")
    print("    - btc_get_prediction")
    print("    - btc_get_consensus")
    print("    - btc_analyze_market")
    print("    - btc_get_model_info")


if __name__ == "__main__":
    main()