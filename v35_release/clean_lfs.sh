#!/bin/sh
# 重写本地历史: 移除所有 LFS 跟踪的数据文件 (只保留代码)
git filter-branch --force --index-filter '
  git ls-files -z | \
  grep -zE "\.(7z|arrow|bz2|ftz|gz|h5|joblib|model|msgpack|onnx|ot|parquet|pb|pt|pth|rar|tflite|tgz|xz|zip|zstandard|db|ark|safetensors|ckpt|gguf|ggml|pt2|mlmodel|npy|npz|pickle|pkl|tar|wasm|zst|json)$|\.bin(\.|$)|\.lfs\.|tfevents|saved_model|llamafile" | \
  xargs -0 -r git rm --cached --ignore-unmatch -q
' -- --all
