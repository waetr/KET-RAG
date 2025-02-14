# Prerequisites
python >= 3.10 is required.

# Installation
```bash
pip install poetry
poetry install
```

# Setup
Take the folder "ragtest-musique" as an example.

1. Initialize the project:
```bash
python -m graphrag init --root ragtest-musique/
```

2. Tune prompts:
```bash
python -m graphrag prompt-tune --root ragtest-musique/ --config ragtest-musique/settings.yaml --discover-entity-types
```

3. Build the index (Note: before building the index, modify the settings.yaml file to fix the params described in the paper as needed)
```bash
python -m graphrag index --root ragtest-musique/
```

# Generate Context and Answers

## Environment Setup
Before running the scripts, set your API key:
```bash
export GRAPHRAG_API_KEY=your_api_key_here
```

## Generate Context
To generate *one* context file, run:
```bash
python indexing_sket/create_context.py ragtest-musique/ keyword 0.5
```

Parameters:
- First argument: root path to your project
- Second argument: strategy for building context ("text", "keyword", or "skeleton")
- Third argument: context threshold theta (0.0-1.0)

## Generate Answers
To generate answers *for all context files* in the output directory, run:
```bash
python indexing_sket/llm_answer.py ragtest-musique/
```

# Acknowledgements
This project is built upon Microsoft's GraphRAG-0.4.1 repository, which is licensed under the MIT license:

https://github.com/microsoft/graphrag/commit/ba50caab4d2fea9bc3fd926dd9051b9f4cebf6bd
