#!/bin/bash
# 跑全部 9 格矩阵 + 跨层一致性（DEVICE 环境变量切换设备）
set -uo pipefail
cd "$(dirname "$0")/.."
exec python3 run.py --all --device "${DEVICE:-p800-kunlunxin}"
