# llama_chat

Llama2 chatbot running locally on M2 Mac based on `Run Llama2 Locally`[^1].
We will also be creating a Langchain[^7] pipeline to support Retrieval Augmented Generation (RAG)[^2][^9].

# Building/Running llama.cpp

```sh
#!/usr/bin/env zsh

# Clone llama.cpp
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp

# Build it. `LLAMA_METAL=1` allows the computation to be executed on the GPU
LLAMA_METAL=1 make
```

## Download model

The original instructions use the GGML[^3] model format but it has been deprecated in Llama.cpp in favor of GGUF[^4].

```sh
export MODEL=llama-2-7b-chat.Q2_K.gguf
if [ ! -f models/${MODEL} ]; then
    curl -L "https://huggingface.co/TheBloke/Llama-2-7b-Chat-GGUF/resolve/main/${MODEL}" -o models/${MODEL}

fi
```

## Setup the default prompt
```sh
export LLM_PROMPT="Hello! How are you?"
```

## Run in interactive mode
```sh
./main \
  -m ./models/llama-2-13b-chat.ggufv3.q4_0.bin \
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

# Setting up python environment for Running Langchain

To be able to run Langchain, we will need to setup a python virtual environment and install all or dependencies.
We will be using Anaconda[^10] since they provide optimized and prebuilt python libraries, but feel free to use
any virtual environment that you wish.

## Create new Conda environment

First we want to create a new Anaconda development envrionment.

### Install from `environment.yml`

```sh
$ export CMAKE_ARGS="-DLLAMA_METAL=on"
$ conda create -n langchain -f environment.yml
$ conda activate langchain
```

### Manual Install

```sh
$ conda create -n langchain python=3.11
$ conda activate langchain
```

Install conda packages for Langchain

```sh
$ conda install -c pytorch pytorch=2.0.1 torchvision torchaudio
$ conda install faiss langchain transformers accelerate einops
```

If we are running on Mac M1/M2, we should make sure that we our pyTourch supports our Metal(`mps`) GPU[^8].

Install conda helper packages

```sh
$ conda install ipython ninja
```

Install remaining pypi dependencies using pip.

```sh
$ pip install bitsandbytes sentence_transformers
```

### Installing `lama-cpp-python` with Metal GPU support[^11]

Build the `llama-cpp-python` library with the `-DLLAMA_METAL=on` CMake Arugument

```sh
$ CMAKE_ARGS="-DLLAMA_METAL=on" pip install -U llama-cpp-python --no-cache-dir
```

We'll also install the server so we can run some tests:

```sh
$ pip install 'llama-cpp-python[server]'
```

Now we can run it with one of our local models.

```sh
$ export MODEL=[path to your llama.cpp gguf models]/[gguf-model-name].bin
$ python3 -m llama_cpp.server --model $MODEL  --n_gpu_layers 1
...
ggml_metal_init: hasUnifiedMemory              = true
ggml_metal_init: recommendedMaxWorkingSetSize  = 49152.00 MB
ggml_metal_init: maxTransferRate               = built-in GPU
llama_new_context_with_model: compute buffer total size =  169.47 MB
llama_new_context_with_model: max tensor size =   102.54 MB
ggml_metal_add_buffer: allocated 'data            ' buffer, size =  3891.95 MB, ( 3892.45 / 49152.00)
ggml_metal_add_buffer: allocated 'eval            ' buffer, size =     1.48 MB, ( 3893.94 / 49152.00)
ggml_metal_add_buffer: allocated 'kv              ' buffer, size =  1026.00 MB, ( 4919.94 / 49152.00)
ggml_metal_add_buffer: allocated 'alloc           ' buffer, size =   168.02 MB, ( 5087.95 / 49152.00)
AVX = 0 | AVX2 = 0 | AVX512 = 0 | AVX512_VBMI = 0 | AVX512_VNNI = 0 | FMA = 0 | NEON = 1 | ARM_FMA = 1 | F16C = 0 | FP16_VA = 0 | WASM_SIMD = 0 | BLAS = 1 | SSE3 = 0 | SSSE3 = 0 | VSX = 0 | 
INFO:     Started server process [22500]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://localhost:8000 (Press CTRL+C to quit)
```

### Using pyTorch with Metal support

`pyTorch` natively supports offloading to the Metal GPU[^12].
You can check by using the following python code:

```python
>>> import torch
>>> torch.backends.mps.is_available()
True
```


# Further Reading

Next try to using some of the prompts from Awesome ChatGPT Prompts[^5] or from FlowGPT[^6]
or creating some of your own.

# References

[^1]: https://replicate.com/blog/run-llama-locally
[^2]: https://medium.com/@murtuza753/using-llama-2-0-faiss-and-langchain-for-question-answering-on-your-own-data-682241488476
[^3]: https://github.com/ggerganov/ggml
[^4]: https://github.com/philpax/ggml/blob/gguf-spec/docs/gguf.md
[^5]: https://github.com/f/awesome-chatgpt-prompts
[^6]: https://flowgpt.com
[^7]: https://docs.langchain.com/
[^8]: https://developer.apple.com/metal/pytorch/
[^9]: https://www.hopsworks.ai/dictionary/retrieval-augmented-generation-llm
[^10]: https://anaconda.org
[^11]: https://github.com/abetlen/llama-cpp-python/blob/main/docs/install/macos.md
[^12]: https://pytorch.org/docs/master/notes/mps.html
