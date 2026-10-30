#!/usr/bin/env bash
set -euo pipefail
python -m jev_agent.cli --mock-llm --mock-jev --controller hybrid
python -m pytest -q