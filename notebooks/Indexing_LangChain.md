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

# Create Vector Index with LangChain

```python
from langchain.embeddings import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
```

## Read the documents

```python
from os import path
from glob import glob


source_path = path.expanduser("~/iCloud/nvAlt/")
all_files = glob(path.join(source_path, "**/*.txt"), recursive=True)
```

```python
from concurrent.futures import ThreadPoolExecutor
from langchain.docstore.document import Document
from langchain.document_loaders import UnstructuredFileLoader
from tqdm import tqdm

def load_document(file_path: str) -> Document:
    return UnstructuredFileLoader(file_path).load()[0]

documents = []
with ThreadPoolExecutor() as executor:
    with tqdm(total=len(all_files), desc="Loading documents") as pbar:
        for document in executor.map(load_document, all_files):
            documents.append(document)
            pbar.update
```

## Create text splitter for our Markdown documents

```python
from langchain.text_splitter import MarkdownHeaderTextSplitter

headers_to_split_on = [
    ("#", "Header1"),
    ("##", "Header2"),
    ("###", "Header3"),
]

markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
```

## Chunk the documents using a text splitter

```python
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    
    chunk_size = 200
    chunk_overlap = 20
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    documents = text_splitter.split_documents(documents)
```

# Langchain PGVector

## Persist Vector Index to PGVector storage

```python
from langchain.vectorstores.pgvector import PGVector

CONNECTION_STRING = "postgresql+psycopg2://mercury:m3ssenger@localhost:5432/nvalt_vector_db"
COLLECTION_NAME = "markdown_notes"

db = PGVector.from_documents(
    embedding=embeddings,
    documents=documents,
    collection_name=COLLECTION_NAME,
    connection_string=CONNECTION_STRING,
)
```

```python
db.similarity_search_with_score("Create a video with ffmpeg.")
```

```python

```
