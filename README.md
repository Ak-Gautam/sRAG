# zRAG

[![Python Package](https://github.com/Ak-Gautam/sRAG/actions/workflows/python-package.yml/badge.svg)](https://github.com/Ak-Gautam/sRAG/actions/workflows/python-package.yml)
[![codecov](https://codecov.io/gh/Ak-Gautam/sRAG/branch/main/graph/badge.svg)](https://codecov.io/gh/Ak-Gautam/sRAG)
[![PyPI version](https://badge.fury.io/py/zrag.svg)](https://badge.fury.io/py/zrag)
[![Python Versions](https://img.shields.io/pypi/pyversions/zrag.svg)](https://pypi.org/project/zrag/)

A lightweight and easy-to-use RAG (Retrieval Augmented Generation) library for building question-answering systems with open-source models and vector stores.

## Installation

```bash
pip install zrag
```

### Development setup

To work on the library and run the test suite locally:

```bash
pip install -e .[dev]
pytest
```

## Quick Start

```python
from zrag import DocumentLoader, RAGPipeline

# Load documents
loader = DocumentLoader()
documents = loader.load("path/to/documents")

# Create RAG pipeline
rag = RAGPipeline()
rag.index(documents)

# Query
response = rag.query("Your question here")
print(response)
```

## Key Features

- **Document Loading**: PDF, text, and markdown support
- **Chunking Strategies**: Token, sentence, and paragraph-based splitting
- **Embeddings**: Support for transformer-based models
- **Vector Stores**: FAISS and ChromaDB integration
- **LLM Integration**: Easy integration with various language models
- **Data Generation**: Synthetic data generation for training/testing

## Components

| Component | Description |
|-----------|-------------|
| `DocumentLoader` | Load and parse documents from various formats |
| `ChunkSplitter` | Split documents into manageable chunks |
| `Embeddings` | Generate vector embeddings for text |
| `VectorStore` | Store and retrieve document embeddings |
| `LLM` | Interface for language model integration |
| `RAGPipeline` | End-to-end RAG implementation |

## Requirements

- Python 3.9+
- PyTorch
- Transformers
- FAISS or ChromaDB
- Additional dependencies in `requirements.txt`

## License

MIT License - see [LICENSE](LICENSE) for details.
