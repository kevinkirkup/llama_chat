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

Create the SQL Alchemy database engine. This provides access to the database instance.

```python

from sqlalchemy import create_engine, MetaData

CONNECTION_STRING = f"postgresql+psycopg2://mercury:m3ssenger@localhost:5432/mercury_dev"

engine = create_engine(CONNECTION_STRING)
```

## Create the ObjectIndex

Next we are going to create an ObjectIndex instance which holds the vector index for our database schema.

```python
from sqlalchemy import MetaData
from llama_index import SQLDatabase, VectorStoreIndex
from llama_index.objects import SQLTableNodeMapping, ObjectIndex, SQLTableSchema

# load all table definitions
metadata_obj = MetaData()
metadata_obj.reflect(engine)

db = SQLDatabase(engine)
table_node_mapping = SQLTableNodeMapping(db)

table_schema_objs = [SQLTableSchema(table_name=table_name) for table_name in metadata_obj.tables.keys()]
    
obj_index = ObjectIndex.from_objects(
    table_schema_objs,
    table_node_mapping,
    VectorStoreIndex,
)
```

## Setup the Service Context

We'll store the vector index in to the service context so that we can use it in future queries.

```python
import os

from llama_index import ServiceContext
from langchain.llms import LlamaCpp

model_path = os.path.expanduser("~/ai/models/llama2/llama-2-70b-chat.Q5_K_M.gguf")
llm = LlamaCpp(
    model_path=model_path,
    n_gpu_layers=1,
    n_batch=512,
    n_ctx=2048,
    f16_kv=True,
    top_p=0.2,
    verbose=True,
    model_kwargs={
        "n_gqa": 8,
    },
)

service_context = ServiceContext.from_defaults(llm=llm)
```

## Create the SQL Query Engine

Setup the query engine with our database, the vector index with our database schema and
the service context which has our LLM.

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

## Ask some questions

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
