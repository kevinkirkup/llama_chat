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


### Initialize Imports

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

```python
sql_model_path = os.path.expanduser("~/ai/models/sqlcoder/sqlcoder.Q5_K_M.gguf")
#sql_model_path = os.path.expanduser("~/ai/models/sqlcoder/sqlcoder.Q5_K_M.gguf")
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

### Create an agent to fetch the users in the database

```python
from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.sql_database import SQLDatabase
from langchain.agents.agent_types import AgentType

db = SQLDatabase.from_uri(
    "postgresql://mercury:m3ssenger@localhost:5432/mercury_dev",
    include_tables=['users', 'roles', 'roles_users', 'orgs', 'ports', 'port_groups', 'virtual_routers', 'links', 'connections'],
    sample_rows_in_table_info=3,
)
toolkit = SQLDatabaseToolkit(
    db=db,
    llm=sql_model,
    verbose=True
)
agent_executor = create_sql_agent(
    llm=model,
    toolkit=toolkit,
    verbose=True,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    handle_parsing_errors=True,
)

```

```python
from langchain.prompts import PromptTemplate

template = PromptTemplate(
    input_variables=["question"],
    template="""
    [INST]
    <<SYS>>
    You are an assistant tasked with querying an Postgresql database for information.
    Given an input question, first create a syntactically correct postgresql query to run,  
    then look at the results of the query and return the answer.
    Only return the requested information from the database.
    
    The valid SQL query tools are:
     1. sql_db_list_tables
     2. sql_db_schema
     3. sql_db_query_checker
     4. sql_db_query
     
    For SQL queries, ALWAYS use the available tools in this order:
     1. sql_db_list_tables
     2. sql_db_schema
     3. sql_db_query_checker
     4. sql_db_query
    <</SYS>>
    
    {question}
    [/INST]
    """,
)

agent_executor.run(template.format(question='How many users have the "org:admin" role?'))
```

```python

```
