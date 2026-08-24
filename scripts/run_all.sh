#!/bin/bash
# 跑全部 6 格矩阵（设备可用 DEVICE 环境变量切换）
set -uo pipefail
cd "$(dirname "$0")/.."
exec python3 run.py --all --device "${DEVICE:-p800-kunlunxin}"
