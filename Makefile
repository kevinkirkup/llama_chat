include MakefileHelp.mk
## ┌──────────────────────────────────────────────────────────────────────────────┐
## │                              Llama2 Chat Makefile                            │
## └──────────────────────────────────────────────────────────────────────────────┘

# Current Build Vars
PARAMETERS?=70b
REPO?=TheBloke/Llama-2-$(PARAMETERS)-Chat-GGUF
MODELS?=llama-2-$(PARAMETERS)-chat.Q4_K_M.gguf \
				llama-2-$(PARAMETERS)-chat.Q5_K_M.gguf

ACTIVE_MODEL?=llama-2-$(PARAMETERS)-chat.Q5_K_M.gguf
LLAMA2_CPP_DIR?=../llama.cpp
LLM_PROMPT?=""

LLM_CONTEXT_SIZE?=2048
LLM_TEMPERATURE?=0.2
LLM_REPEAT_PENALTY?=1.1
LLM_TOP_K?=10000
LLM_N_PRODICT?=-1
LLM_BATCH_SIZE?=256
LLM_THREADS?=8


install: ## Download Models from Hugging Face
	@./bin/download_models.sh $(REPO) $(MODELS)
.PHONY: install

run_llma_cpp: install ## Run the active Model with llama.cpp directly
	@$(LLAMA2_CPP_DIR)/main \
  	--model ${MODEL_DIR}/llama2/${ACTIVE_MODEL} \
  	--prompt "$(LLM_PROMPT)" \
  	--color \
  	--ctx_size $(LLM_CONTEXT_SIZE) \
  	--n-predict $(LLM_N_PRODICT) \
  	--instruct \
  	--batch-size $(LLM_BATCH_SIZE) \
  	--top_k $(LLM_TOP_K) \
  	--temp $(LLM_TEMPERATURE) \
  	--repeat_penalty $(LLM_REPEAT_PENALTY) \
  	-t $(LLM_THREADS)
.PHONY: run_llma_cpp

run: install ## Run the Llama_cpp HTTP Server
	@python3 -m llama_cpp.server \
		--model ${MODEL_DIR}/llama2/${ACTIVE_MODEL} \
		--n_gpu_layers 100
