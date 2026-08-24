#!/bin/bash
exec python3 "$(dirname "$0")/../run.py" --route a1 --level framework --device "${DEVICE:-p800-kunlunxin}" "$@"
