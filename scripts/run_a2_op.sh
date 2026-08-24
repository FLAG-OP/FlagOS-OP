#!/bin/bash
exec python3 "$(dirname "$0")/../run.py" --route a2 --level op --device "${DEVICE:-p800-kunlunxin}" "$@"
