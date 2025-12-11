#!/usr/bin/env python3
"""
Monitor advanced ML training progress
"""

import os
import time
import psutil
from datetime import datetime

def monitor_training():
    """Monitor the training process"""
    # Find training process
    training_pid = None
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            cmdline = proc.info.get('cmdline', [])
            if cmdline and 'train_advanced_ml_models.py' in ' '.join(cmdline):
                training_pid = proc.info['pid']
                break
        except:
            continue

    if not training_pid:
        print("❌ Training process not found")
        return

    print(f"✅ Found training process: PID {training_pid}")

    try:
        proc = psutil.Process(training_pid)

        # Process info
        create_time = datetime.fromtimestamp(proc.create_time())
        runtime = datetime.now() - create_time

        print(f"⏱️  Runtime: {runtime}")
        print(f"💾 Memory: {proc.memory_info().rss / 1024 / 1024:.1f} MB")
        print(f"🔥 CPU: {proc.cpu_percent(interval=1):.1f}%")

        # Check if models directory exists
        models_dir = "models/"
        if os.path.exists(models_dir):
            # List recent model files
            import glob
            model_files = glob.glob(f"{models_dir}/*.pkl")
            if model_files:
                print(f"\n📦 Model files found:")
                for mf in sorted(model_files, key=os.path.getmtime)[-5:]:
                    mod_time = datetime.fromtimestamp(os.path.getmtime(mf))
                    size_mb = os.path.getsize(mf) / 1024 / 1024
                    print(f"  - {os.path.basename(mf)}: {size_mb:.1f}MB ({mod_time.strftime('%H:%M:%S')})")

        # Estimate progress (based on typical runtime)
        if runtime.total_seconds() < 300:
            print(f"\n📊 Estimated progress: Data collection phase")
        elif runtime.total_seconds() < 600:
            print(f"\n📊 Estimated progress: Feature engineering phase")
        elif runtime.total_seconds() < 900:
            print(f"\n📊 Estimated progress: Model training phase (XGBoost/LightGBM)")
        else:
            print(f"\n📊 Estimated progress: Advanced model training (Neural Net/Transformer)")

        print(f"\n⏳ Training still in progress...")
        print(f"   Advanced ML training with Transformer can take 15-30 minutes")

    except psutil.NoSuchProcess:
        print("❌ Training process ended")
    except Exception as e:
        print(f"❌ Error monitoring: {e}")

if __name__ == "__main__":
    monitor_training()