#!/usr/bin/env python3
"""
BTC Trading System - 실행 스크립트
"""

import sys
import os

# 시스템 경로 추가
sys.path.append(os.path.join(os.path.dirname(__file__), 'core'))

# 메인 모듈 import
from main import main

if __name__ == "__main__":
    main()