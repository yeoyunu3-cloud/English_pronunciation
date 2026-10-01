#!/bin/zsh
# 스피치 발음 카드 실행: 처음 한 번만 환경을 만들고 모델을 내려받은 뒤 서버를 켭니다.
cd "$(dirname "$0")"
if [ ! -x .venv/bin/python ]; then
  uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -r requirements.txt || exit 1
fi
mkdir -p models
BASE=https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0
[ -f models/voices-v1.0.bin ] || curl -fL -C - -o models/voices-v1.0.bin $BASE/voices-v1.0.bin
[ -f models/kokoro-v1.0.onnx ] || curl -fL -C - -o models/kokoro-v1.0.onnx $BASE/kokoro-v1.0.onnx
(sleep 4 && open http://localhost:8000) &
exec .venv/bin/python server.py
