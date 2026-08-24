#!/bin/bash
exec python3 "$(dirname "$0")/../run.py" --route a2 --level framework --device "${DEVICE:-p800-kunlunxin}" "$@"
