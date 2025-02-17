# KET-RAG: Knowledge-Enhanced Text Retrieval Augmented Generation

<div align="center"> 
<table border="0" width="100%">
<tr>
<td>
<div>
    <p style="text-align: center;">
        <a href='https://github.com/waetr/KET-RAG/'><img src='https://img.shields.io/badge/Project-Page-Green'></a>
        <a href='https://arxiv.org/abs/2502.09304'><img src='https://img.shields.io/badge/arXiv-2502.09304-b31b1b'></a>
        <img src="https://img.shields.io/badge/python-3.10-blue">
    </p>
    <p style="text-align: center;">
        <img src="https://raw.githubusercontent.com/waetr/KET-RAG/main/ketrag.png" alt="KET-RAG Image" width="width: 70%; max-width: 800px;">
    </p>
</div>
</td>
</tr>
</table>
</div>

**KET-RAG** is a powerful and flexible framework for retrieval-augmented generation (RAG) enhanced with knowledge graphs. This project allows for structured document indexing and efficient LLM-based answer generation.

## Overview

KET-RAG balances retrieval quality and efficiency with a multi-granular indexing framework consisting of:

- Knowledge Graph Skeleton (*SkeletonRAG*): Selects key text chunks via PageRank and extracts structured knowledge using LLMs.
- Text-Keyword Bipartite Graph (*KeywordRAG*): Links keywords to text chunks, mimicking knowledge graph relationships with minimal cost.

During retrieval, KET-RAG integrates information from both entity and keyword channels, enabling efficient and high-quality LLM-based answer generation. Experiments show that KET-RAG significantly reduces indexing costs while improving retrieval and generation quality, making it a practical solution for large-scale RAG applications.

## Prerequisites

Ensure you have Python **>=3.10** installed.

## Installation

Install dependencies using Poetry:

```bash
pip install poetry
poetry install
```

## Setup

Using the folder `ragtest-musique` as an example, follow these steps:

### 1. Initialize the Project

```bash
python -m graphrag init --root ragtest-musique/
```

This command sets up the necessary file structure and configurations.

### 2. Tune Prompts

```bash
python -m graphrag prompt-tune --root ragtest-musique/ --config ragtest-musique/settings.yaml --discover-entity-types
```

Adjust prompts for better retrieval.

### 3. Build the Index

Before running this step, modify `settings.yaml` to set the appropriate parameters as needed, based on our paper.

```bash
python -m graphrag index --root ragtest-musique/
```

This process creates an indexed structure for retrieval.

## Generate Context and Answers

### Environment Setup

Before executing the scripts, set up your API key:

```bash
export GRAPHRAG_API_KEY=your_api_key_here
```

### 1. Generate Context

To generate a single context file:

```bash
python indexing_sket/create_context.py ragtest-musique/ keyword 0.5
```

#### Parameters:

- **First argument**: Root directory of the project
- **Second argument**: Context-building strategy (`text`, `keyword`, or `skeleton`)
- **Third argument**: Context threshold **theta** (range: `0.0-1.0`)

### 2. Generate Answers

To generate answers for all context files in the output directory:

```bash
python indexing_sket/llm_answer.py ragtest-musique/
```

## Acknowledgments

This project builds upon [Microsoft's GraphRAG (version 0.4.1)](https://github.com/microsoft/graphrag/commit/ba50caab4d2fea9bc3fd926dd9051b9f4cebf6bd), licensed under the MIT License.

For more details, read our paper on [arXiv](https://arxiv.org/abs/2502.09304).
