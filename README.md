# llama_chat
Llama2 chatbot running locally on M2 Mac based on `Run Llama2 Locally`[^1].

# Build llama.cpp

```sh
#!/usr/bin/env zsh

# Clone llama.cpp
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp

# Build it. `LLAMA_METAL=1` allows the computation to be executed on the GPU
LLAMA_METAL=1 make
```

# Download model

The original instructions use the GGML[^2] model format but it has been deprecated in Llama.cpp in favor of GGUF[^3].

```sh
export MODEL=llama-2-7b-chat.Q2_K.gguf
if [ ! -f models/${MODEL} ]; then
    curl -L "https://huggingface.co/TheBloke/Llama-2-7b-Chat-GGUF/resolve/main/${MODEL}" -o models/${MODEL}

fi
```

# Setup the default prompt
```sh
export LLM_PROMPT="Hello! How are you?"
```

# Run in interactive mode
```sh
./main \
  -m ./models/llama-2-13b-chat.ggmlv3.q4_0.bin \
  -p "${LLM_PROMPT}" \
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

[^1]: (https://replicate.com/blog/run-llama-locally)
[^2]: https://github.com/ggerganov/ggml
[^3]: https://github.com/philpax/ggml/blob/gguf-spec/docs/gguf.md
