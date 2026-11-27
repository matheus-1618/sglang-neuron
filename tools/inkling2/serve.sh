#!/bin/bash
# Inkling-2-Mini on SGLang Neuron (port branch inkling2/port), TP8 LNC1, 256-token pages, chunk 256; clients inside
# the container only (127.0.0.1). RADIX=1 keeps SGLang's radix cache on (default: off).
MODEL=${MODEL:-/models/Inkling-2-Mini-text}
export SGLANG_PLATFORM=neuron NEURON_RT_ENABLE_DEVICE_MEMORY_OOM_LOG=${OOMLOG:-0}
# the platform refuses a NEFF cache built from other kernel or model sources: the directory is keyed on them
SRC=/usr/local/lib/python3.14/site-packages/sglang_neuron
KEY=$(cat $SRC/models/inkling2/*.py $SRC/models/inkling2/kernels/*.py $SRC/models/catalog.py 2>/dev/null | md5sum | cut -c1-10)
CACHE_DIR=${CACHE_DIR:-/cache/inkl_sgl_cache_$KEY}
echo "compile cache $CACHE_DIR"
export TORCH_NEURONX_NEFF_CACHE_DIR=$CACHE_DIR/neff TORCH_NEURONX_HLO_CACHE_DIR=$CACHE_DIR/hlo NKI_COMPILE_CACHE_URL=$CACHE_DIR/nki
if [ -n "${RADIX:-}" ]; then CACHE_FLAGS=""; else CACHE_FLAGS="--disable-radix-cache"; fi
exec python3 -m sglang.launch_server --model-path $MODEL --tp-size 8 --context-length ${CTX:-16384} --page-size 256 \
  --chunked-prefill-size ${CHUNK:-256} --max-running-requests ${MRR:-2} --mem-fraction-static ${MFS:-0.85} --max-total-tokens ${MTT:-32768} --cuda-graph-bs-decode ${CGBS:-2} \
  $CACHE_FLAGS --dtype bfloat16 --host 127.0.0.1 --port ${PORT:-30000} ${EXTRA:-}
