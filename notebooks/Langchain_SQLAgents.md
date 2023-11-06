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

# Langchain Agent testing

## Setup

### Initial Imports

```python
import asyncio
import os

from langchain.chains import LLMChain
```

### Initialize the Llama2 Model

```python
from langchain.llms import LlamaCpp

model_path = os.path.expanduser("~/ai/models/llama2/llama-2-70b-chat.Q5_K_M.gguf")
model = LlamaCpp(
    model_path=model_path,
    n_gpu_layers=1,
    n_batch=512,
    n_ctx=2048,
    f16_kv=True,
    temporature=0,
    n_gqa=8,
    verbose=True
)
```

### Initialize the SQL Coder model

```python
sql_model_path = os.path.expanduser("~/ai/models/llama2/llama-2-7b-chat.Q5_K_M.gguf")
sql_model = LlamaCpp(
    model_path=sql_model_path,
    n_gpu_layers=1,
    n_batch=512,
    n_ctx=2048,
    f16_kv=True,
    temporature=0,
    n_gqa=8,
    verbose=True
)
```

### Create a connection to the SQL Database

```python
from langchain.sql_database import SQLDatabase

db = SQLDatabase.from_uri(
    "postgresql://mercury:m3ssenger@localhost:5432/mercury_dev",
    include_tables=[
        'users',
        'roles',
        'roles_users',
        'orgs',
        'ports',
        'port_groups',
        'virtual_routers',
        'links',
        'connections'
    ],
    sample_rows_in_table_info=3,
)

```

### Create The SQL Toolkit we'll use with our Agent

```python
from textwrap import dedent

from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.agents.agent_types import AgentType

toolkit = SQLDatabaseToolkit(
    db=db,
    llm=sql_model,
    verbose=True
)

prefix = dedent('''
    <<SYS>>
    You are an agent designed to interact with a SQL database.
    Given an input question, create a syntactically correct {dialect} query to run, then look at the results of the query and return the answer.
    Unless the user specifies a specific number of examples they wish to obtain, always limit your query to at most {top_k} results.
    You can order the results by a relevant column to return the most interesting examples in the database.
    Never query for all the columns from a specific table, only ask for the relevant columns given the question.
    You have access to tools for interacting with the database.
    Only use the below tools. Only use the information returned by the below tools to construct your final answer.
     
    For SQL queries, ALWAYS use the available tools in this order:
     1. sql_db_list_tables
     2. sql_db_schema
     3. sql_db_query_checker
     4. sql_db_query
     
    You MUST double check your query before executing it. If you get an error while executing a query, rewrite the query and try again.
    
    DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.
    
    If the question does not seem related to the database, just return "I don\'t know" as the answer.

    <</SYS>>''').strip()

agent_executor = create_sql_agent(
    llm=model,
    toolkit=toolkit,
    prefix=prefix,
    verbose=True,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    handle_parsing_errors=True
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

agent_executor.run(template.format(question='How many users have the "org:admin" role?'))
```

```python

```
