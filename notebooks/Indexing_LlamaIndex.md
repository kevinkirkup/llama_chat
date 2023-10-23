---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.15.2
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

# Postgres RAG

## Define the  Service Context

```python
import os

from llama_index import SimpleDirectoryReader

input_path = os.path.expanduser("~/iCloud/nvAlt/")
# documents_path = os.path.expanduser("~/Desktop/tmp/nvAlt/")
documents = SimpleDirectoryReader(
    input_dir=input_path,
    exclude_hidden=True,
    exclude=["Notes & Settings"]
).load_data()
```

```python
from llama_index import (
  LangchainEmbedding,
  ServiceContext,
  VectorStoreIndex,
  set_global_service_context,
)
from llama_index.llms import LlamaCPP
from llama_index.llms.llama_utils import (
    messages_to_prompt,
    completion_to_prompt,
)

from langchain.embeddings.huggingface import HuggingFaceEmbeddings

model_path = os.path.expanduser("~/ai/models/llama2/llama-2-13b-chat.Q5_K_M.gguf")
llm = LlamaCPP(
    model_path=model_path,
    temperature=0.1,
    max_new_tokens=256,
    # llama2 has a context window of 4096 tokens
    context_window=4096,
    # kwargs to pass to __call__()
    generate_kwargs={},
    # kwargs to pass to __init__()
    # set to at least 1 to use GPU
    model_kwargs={"n_gpu_layers": 1},
    # transform inputs into Llama2 format
    messages_to_prompt=messages_to_prompt,
    completion_to_prompt=completion_to_prompt,
    verbose=True,
)

embed_model = LangchainEmbedding(
  HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
)

service_context = ServiceContext.from_defaults(
    llm=llm,
    embed_model=embed_model,
)

set_global_service_context(service_context)
```

## Create the Vector Database


### Make sure we set the PGVector Vector Size since we are using HuggingFaceEmbedding

https://github.com/langchain-ai/langchain/pull/3964

```python
os.environ["PGVECTOR_VECTOR_SIZE"] = "768"
```

```python
import psycopg2

from llama_index.vector_stores import PGVectorStore
from llama_index import StorageContext

CONNECTION_STRING = "postgresql://mercury:m3ssenger@localhost:5432/postgres"
DATABASE_NAME = "nvalt_vector_db"

conn = psycopg2.connect(CONNECTION_STRING)
conn.autocommit = True

with conn.cursor() as c:
    c.execute(f"DROP DATABASE IF EXISTS {DATABASE_NAME}")
    c.execute(f"CREATE DATABASE {DATABASE_NAME}")
```

## Create the Vector Storage context

```python
from sqlalchemy import make_url

url = make_url(CONNECTION_STRING)
vector_store = PGVectorStore.from_params(
    database=DATABASE_NAME,
    host=url.host,
    password=url.password,
    port=url.port,
    user=url.username,
    table_name="vector_data",
    embed_dim=768,  # openai embedding dimension
)

storage_context = StorageContext.from_defaults(
    vector_store=vector_store
)
```

# Create Tools for extracting Text

## Create the Metadata Extractors

```python
from llama_index.node_parser import SimpleNodeParser
from llama_index.node_parser.extractors import (
    KeywordExtractor,
    MetadataExtractor,
    TitleExtractor,
    SummaryExtractor,
    QuestionsAnsweredExtractor
)
from llama_index.text_splitter import TokenTextSplitter

model_path = os.path.expanduser("~/ai/models/llama2/llama-2-13b-chat.Q5_K_M.gguf")
summary_llm = LlamaCPP(
    model_path=model_path,
    temperature=0.1,
    max_new_tokens=256,
    # llama2 has a context window of 4096 tokens
    context_window=4096,
    # kwargs to pass to __call__()
    generate_kwargs={},
    # kwargs to pass to __init__()
    # set to at least 1 to use GPU
    model_kwargs={"n_gpu_layers": 1},
    # transform inputs into Llama2 format
    messages_to_prompt=messages_to_prompt,
    completion_to_prompt=completion_to_prompt,
    verbose=True,
)

metadata_extractor = MetadataExtractor(
    extractors=[
        TitleExtractor(nodes=5),
        # QuestionsAnsweredExtractor(questions=3),
        SummaryExtractor(llm=summary_llm),
        KeywordExtractor(keywords=5),
    ],
)
text_splitter = TokenTextSplitter(separator=" ", chunk_size=512, chunk_overlap=128)

node_parser = SimpleNodeParser.from_defaults(
    metadata_extractor=metadata_extractor,
    text_splitter=text_splitter,
)
```

## Parse the doucuments we want to index

```python
nodes = node_parser.get_nodes_from_documents(
    documents=documents,
    show_progress=True,
)
```

# Create Vector index


### Create the vectors for our Nodes and store them in the database

```python
for idx, node in enumerate(nodes):
    if "\x00" in node.text:
        print(f"Found in node {idx}")

```

```python
from llama_index import VectorStoreIndex

index = VectorStoreIndex(
    nodes,
    service_context=service_context,
    storage_context=storage_context,
    show_progress=True,
)
```
```python
query_engine = index.as_query_engine()
```

```python
response = query_engine.query("Where can I find a command on how to encode video with ffmpeg?")
```

```python
import textwrap

print(textwrap.fill(str(response), 100))
```

```python

```
