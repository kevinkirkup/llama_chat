#!/usr/bin/env sh
##################################################
# Download Hugging Face Models
##################################################

REPO=${1}
MODELS=${@:2}

for MODEL in $MODELS; do
  if [ ! -f ${MODEL_DIR}/llama2/${MODEL} ]; then

    echo Downloading ${MODEL} from ${REPO} ...
    curl -L "https://huggingface.co/${REPO}/resolve/main/${MODEL}" -o ${MODEL_DIR}/llama2/${MODEL}
    echo done.

  else
    echo Model ${MODEL} from ${REPO} already exists!
  fi
done
