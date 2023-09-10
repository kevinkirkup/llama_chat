# llama_chat
Llama2 chatbot running locally on M2 Mac

# Build llama.cpp

```sh
#!/bin/bash

# Clone llama.cpp
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp

# Build it. `LLAMA_METAL=1` allows the computation to be executed on the GPU
LLAMA_METAL=1 make
```

# Download model
```sh
export MODEL=llama-2-7b-chat.Q2_K.gguf
if [ ! -f models/${MODEL} ]; then
    curl -L "https://huggingface.co/TheBloke/Llama-2-7b-Chat-GGUF/resolve/main/${MODEL}" -o models/${MODEL}

fi
```

# Setup the default prompt
```sh
PROMPT="Hello! How are you?"
```

# Run in interactive mode
```sh
./main -m ./models/llama-2-13b-chat.ggmlv3.q4_0.bin \
  --color \
  --ctx_size 2048 \
  -n -1 \
  -ins -b 256 \
  --top_k 10000 \
  --temp 0.2 \
  --repeat_penalty 1.1 \
  -t 8
```

# References

* [Run Llama2 Locally](https://replicate.com/blog/run-llama-locally)
