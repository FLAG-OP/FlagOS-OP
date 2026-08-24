#!/bin/bash
exec python3 "$(dirname "$0")/../run.py" --route b --level op --device "${DEVICE:-p800-kunlunxin}" "$@"
