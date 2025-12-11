#!/usr/bin/env python3
"""
Test script for real confidence calculation
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from btc_predictor import BTCPredictor
import json

def test_real_confidence():
    """Test real confidence calculation (prediction probability × model accuracy)"""
    predictor = BTCPredictor()

    print("=" * 80)
    print("🧮 REAL CONFIDENCE CALCULATION TEST")
    print("Real Confidence = Prediction Probability × Model Accuracy")
    print("=" * 80)

    # Test individual predictions
    print("\n📊 Individual Model Analysis:")
    print("-" * 80)
    print(f"{'Model':<15} {'Prediction':<12} {'Model Acc':<12} {'Real Conf':<12} {'Interpretation'}")
    print("-" * 80)

    models = [
        ('1h', 'UP'),
        ('1h', 'DOWN'),
        ('4h', 'UP'),
        ('4h', 'DOWN'),
        ('30m', 'UP'),
        ('30m', 'DOWN'),
        ('15m', 'UP'),
    ]

    for timeframe, direction in models:
        pred = predictor.predict(timeframe, direction)

        if 'error' not in pred:
            confidence = pred['confidence']
            accuracy = pred['model_accuracy']
            real_conf = pred['real_confidence']

            # Interpretation
            if real_conf > 0.6:
                interpretation = "✅ Strong (>60%)"
            elif real_conf > 0.5:
                interpretation = "🟡 Moderate (50-60%)"
            elif real_conf > 0.4:
                interpretation = "🟠 Weak (40-50%)"
            else:
                interpretation = "⚪ Very Weak (<40%)"

            print(f"{timeframe:>2} {direction:<12} {confidence:.1%} × {accuracy:.1f}% = {real_conf:.1%}    {interpretation}")

    # Test consensus
    print("\n" + "=" * 80)
    print("📊 CONSENSUS ANALYSIS")
    print("=" * 80)

    consensus = predictor.get_consensus()

    print(f"\n🎯 Consensus Direction: {consensus['consensus']}")
    print(f"\nPrediction-based scores:")
    print(f"  UP Score: {consensus['up_score']:.1%}")
    print(f"  DOWN Score: {consensus['down_score']:.1%}")
    print(f"  Confidence: {consensus['confidence']:.1%}")

    print(f"\n⭐ Real confidence scores (accounting for model accuracy):")
    print(f"  Real UP Score: {consensus['real_up_score']:.1%}")
    print(f"  Real DOWN Score: {consensus['real_down_score']:.1%}")
    print(f"  Real Consensus Confidence: {consensus['real_confidence']:.1%}")

    print(f"\n📈 Active Signals:")
    for signal in consensus['active_signals']:
        print(f"  • {signal}")

    # Analysis
    print("\n" + "=" * 80)
    print("💡 INSIGHT ANALYSIS")
    print("=" * 80)

    print("\n🔍 What Real Confidence Tells Us:")
    print("• Real confidence combines current prediction strength with historical accuracy")
    print("• It represents the actual probability that this prediction will be correct")
    print("• Values above 60% are considered strong signals for trading")
    print("• Values below 40% should be treated with caution")

    print("\n📊 Current Market Situation:")
    real_conf = consensus['real_confidence']
    if real_conf > 0.6:
        print(f"✅ Strong signal with {real_conf:.1%} actual success probability")
    elif real_conf > 0.5:
        print(f"🟡 Moderate signal with {real_conf:.1%} actual success probability")
    elif real_conf > 0.4:
        print(f"🟠 Weak signal with {real_conf:.1%} actual success probability")
    else:
        print(f"⚪ No clear signal with only {real_conf:.1%} success probability")

    print("\n" + "=" * 80)
    print("✅ TEST COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    test_real_confidence()