#!/usr/bin/env bash
set -euo pipefail
printf 'AndroidWorld host preflight\n'
python --version
for cmd in adb emulator java; do command -v "$cmd" || true; done
ls -l /dev/kvm 2>/dev/null || echo '/dev/kvm absent (emulation can be slow/unavailable)'
if command -v adb >/dev/null 2>&1; then adb devices -l; fi
python - <<'PY'
try:
    import android_world
    print('android_world import OK:', android_world.__file__)
except ImportError as exc:
    print('android_world NOT INSTALLED:', exc)
PY