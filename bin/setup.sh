#!/usr/bin/env zsh
# Setup LLM configurations

##################################################
# Llama 2
##################################################
# Build it. `LLAMA_METAL=1` allows the computation to be executed on the GPU
export LLAMA_METAL=1

export MODELS=(
llama-2-7b-chat.Q2_K.gguf
)

# The model
function download_models() {
  for MODEL in $MODELS; do
    if [ ! -f ${MODEL_DIR}/${MODEL} ]; then
        curl -L "https://huggingface.co/TheBloke/Llama-2-7b-Chat-GGUF/resolve/main/${MODEL}" -o ${MODEL_DIR}/${MODEL}
    fi
  done
}

function run_model() {
  local MODEL=$1

  ./main -m ${MODEL_DIR}/${MODEL} \
    --color \
    --ctx_size 2048 \
    -n -1 \
    -ins -b 256 \
    --top_k 10000 \
    --temp 0.2 \
    --repeat_penalty 1.1 \
    -t 8
}

