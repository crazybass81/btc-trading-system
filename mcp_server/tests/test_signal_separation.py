#!/usr/bin/env python3
"""
Test script for verifying UP/DOWN signal separation
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from btc_predictor import BTCPredictor
import json

def test_signal_separation():
    """Test that UP models only give UP signals and DOWN models only give DOWN signals"""
    predictor = BTCPredictor()

    print("=" * 60)
    print("🧪 TESTING SIGNAL SEPARATION")
    print("=" * 60)

    # Test all models
    test_cases = [
        ('1h', 'UP'),
        ('1h', 'DOWN'),
        ('4h', 'UP'),
        ('4h', 'DOWN'),
        ('30m', 'UP'),
        ('30m', 'DOWN'),
        ('15m', 'UP'),
    ]

    results = []

    for timeframe, direction in test_cases:
        print(f"\n📊 Testing {timeframe} {direction} Model:")
        print("-" * 40)

        prediction = predictor.predict(timeframe, direction)

        if 'error' in prediction:
            print(f"❌ Error: {prediction['error']}")
            continue

        signal = prediction.get('signal', 'UNKNOWN')
        signal_strength = prediction.get('signal_strength', 0)
        confidence = prediction.get('confidence', 0)

        # Check if signal is correct
        is_correct = False
        if direction == 'UP':
            is_correct = signal in ['UP', 'NO_SIGNAL']
            if signal == 'DOWN':
                print(f"⚠️ ERROR: UP model generated DOWN signal!")
        else:  # DOWN
            is_correct = signal in ['DOWN', 'NO_SIGNAL']
            if signal == 'UP':
                print(f"⚠️ ERROR: DOWN model generated UP signal!")

        status_emoji = "✅" if is_correct else "❌"

        print(f"{status_emoji} Model: {timeframe} {direction}")
        print(f"   Signal: {signal}")
        print(f"   Signal Strength: {signal_strength:.1%}")
        print(f"   Confidence: {confidence:.1%}")
        print(f"   Model Accuracy: {prediction['model_accuracy']}%")

        results.append({
            'model': f"{timeframe}_{direction}",
            'signal': signal,
            'correct': is_correct,
            'signal_strength': signal_strength,
            'confidence': confidence
        })

    # Test consensus
    print("\n" + "=" * 60)
    print("📊 TESTING CONSENSUS")
    print("=" * 60)

    consensus = predictor.get_consensus()

    print(f"\nConsensus: {consensus['consensus']}")
    print(f"Confidence: {consensus['confidence']:.1%}")
    print(f"UP Score: {consensus.get('up_score', 0):.1%}")
    print(f"DOWN Score: {consensus.get('down_score', 0):.1%}")
    print(f"Active Signals: {', '.join(consensus.get('active_signals', [])) if consensus.get('active_signals') else 'None'}")

    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)

    correct_count = sum(1 for r in results if r['correct'])
    total_count = len(results)

    print(f"\n✅ Passed: {correct_count}/{total_count}")
    print(f"❌ Failed: {total_count - correct_count}/{total_count}")

    if correct_count == total_count:
        print("\n🎉 ALL TESTS PASSED! Models correctly separate UP/DOWN signals")
    else:
        print("\n⚠️ Some tests failed. Please review the results above.")

    # Show active signals only
    print("\n" + "=" * 60)
    print("🎯 ACTIVE SIGNALS ONLY")
    print("=" * 60)

    active_signals = [r for r in results if r['signal'] != 'NO_SIGNAL']
    if active_signals:
        for result in active_signals:
            emoji = "📈" if 'UP' in result['signal'] else "📉"
            print(f"{emoji} {result['model']}: {result['signal']} ({result['signal_strength']:.1%})")
    else:
        print("⏸️ No active signals at this time")

    return correct_count == total_count


if __name__ == "__main__":
    success = test_signal_separation()
    exit(0 if success else 1)