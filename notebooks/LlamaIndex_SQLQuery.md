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

# LlamaIndex for querying SQL
In this notebook, we'll trying using LlamaIndex for generating indices for an SQL Database and compare it's performance against LangChains SQLAgent.


## Create the Database Engine

```python

from sqlalchemy import create_engine, MetaData

CONNECTION_STRING = f"postgresql+psycopg2://mercury:m3ssenger@localhost:5432/mercury_dev"

engine = create_engine(CONNECTION_STRING)
```

## Create the ObjectIndex

```python
from sqlalchemy import MetaData
from llama_index import SQLDatabase, VectorStoreIndex
from llama_index.objects import SQLTableNodeMapping, ObjectIndex, SQLTableSchema

# load all table definitions
metadata_obj = MetaData()
metadata_obj.reflect(engine)

db = SQLDatabase(engine)
table_node_mapping = SQLTableNodeMapping(db)

table_schema_objs = []
for table_name in metadata_obj.tables.keys():
    table_schema_objs.append(SQLTableSchema(table_name=table_name))
    
# We dump the table schema information into a vector index. The vector index is stored within the context builder for future use.
obj_index = ObjectIndex.from_objects(
    table_schema_objs,
    table_node_mapping,
    VectorStoreIndex,
)
```

```python
import os

from llama_index import LLMPredictor, ServiceContext
from langchain.llms import LlamaCpp

model_path = os.path.expanduser("~/ai/models/llama2/llama-2-70b-chat.Q5_K_M.gguf")
llm = LlamaCpp(
    model_path=model_path,
    n_gpu_layers=1,
    n_batch=512,
    n_ctx=2048,
    f16_kv=True,
    temporature=0,
    n_gqa=8,
    verbose=True
)

llm_predictor = LLMPredictor(llm=llm)
service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor)
```

```python
from llama_index.indices.struct_store import SQLTableRetrieverQueryEngine

# We construct a SQLTableRetrieverQueryEngine. 
# Note that we pass in the ObjectRetriever so that we can dynamically retrieve the table during query-time.
# ObjectRetriever: A retriever that retrieves a set of query engine tools.
query_engine = SQLTableRetrieverQueryEngine(
    db,
    obj_index.as_retriever(similarity_top_k=1),
    service_context=service_context,
)
```

```python
from langchain.prompts import PromptTemplate

template = PromptTemplate(
    input_variables=["question"],
    template="""
    [INST]
    {question}
    [/INST]
    """,
)

response = query_engine.query(template.format(question='How many users have the "org:admin" role?'))
print(response)
print(response.metadata['sql_query'])
print(response.metadata['result'])
```

```python

```
