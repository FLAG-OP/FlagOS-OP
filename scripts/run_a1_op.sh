#!/bin/bash
exec python3 "$(dirname "$0")/../run.py" --route a1 --level op --device "${DEVICE:-p800-kunlunxin}" "$@"
