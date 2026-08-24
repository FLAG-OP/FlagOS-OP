#!/bin/bash
exec python3 "$(dirname "$0")/../run.py" --route b --level framework --device "${DEVICE:-p800-kunlunxin}" "$@"
