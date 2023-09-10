include MakefileHelp.mk
## ┌──────────────────────────────────────────────────────────────────────────────┐
## │                              Llama2 Chat Makefile                            │
## └──────────────────────────────────────────────────────────────────────────────┘

# Current Build Vars
REPO?=TheBloke/Llama-2-7b-Chat-GGUF
MODELS?=llama-2-7b-chat.Q4_K_M.gguf \
				llama-2-7b-chat.Q5_K_M.gguf
ACTIVE_MODEL?=llama-2-7b-chat.Q5_K_M.gguf
LLAMA2_CPP_DIR?=../llama.cpp
LLM_PROMPT?=""

download: ## Download Models from Hugging Face
	@./bin/download_models.sh $(REPO) $(MODELS)
.PHONY: download

run: download ## Run the active Model
	@$(LLAMA2_CPP_DIR)/main \
  	-m ${MODEL_DIR}/llama2/${ACTIVE_MODEL} \
  	-p "$(LLM_PROMPT)" \
  	--color \
  	--ctx_size 2048 \
  	-n -1 \
  	-ins -b 256 \
  	--top_k 10000 \
  	--temp 0.2 \
  	--repeat_penalty 1.1 \
  	-t 8
.PHONY: run
