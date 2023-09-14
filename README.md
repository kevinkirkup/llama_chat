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

# Setting up python environment for Running Langchain

To be able to run Langchain, we will need to setup a python virtual environment and install all or dependencies.
We will be using Anaconda[^10] since they provide optimized and prebuilt python libraries, but feel free to use
any virtual environment that you wish.

## Create new Conda environment

Create a new Anaconda virtual environment.

```sh
$ conda create -n langchain python=3.11
$ conda activate langchain
```

Install conda packages for Langchain

```sh
$ conda install -c pytorch pytorch=2.0.1 torchvision torchaudio
$ conda install faiss langchain transformers accelerate einops
```

Install conda helper packages

```sh
$ conda install ipython ninja
```

Install remaining pypi dependencies using pip.

```sh
$ pip install bitsandbytes sentence_transformers
```

If we are running on Mac M1/M2, we should make sure that we our pyTourch supports our Metal(`mps`) GPU[^8].

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
