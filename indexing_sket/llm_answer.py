import asyncio
import copy
import math
import random
import sys
import json
import os
import glob
import re
import time
from util_v1 import MyLocalSearch
from itertools import product

import pandas as pd
import tiktoken
from collections import Counter, defaultdict
from tqdm.asyncio import tqdm_asyncio

from graphrag.query.context_builder.entity_extraction import (
    EntityVectorStoreKey,
)
from graphrag.query.input.loaders.dfs import (
    store_entity_semantic_embeddings,
)
from graphrag.query.llm.oai.chat_openai import ChatOpenAI
from graphrag.query.llm.oai.embedding import OpenAIEmbedding
from graphrag.query.llm.oai.typing import OpenaiApiType
from graphrag.query.question_gen.local_gen import LocalQuestionGen
from graphrag.query.structured_search.local_search.mixed_context import (
    LocalSearchMixedContext,
)
from graphrag.query.structured_search.local_search.search import LocalSearch
from graphrag.vector_stores.lancedb import LanceDBVectorStore


async def process_qa_pairs(search_engine, golden, graphgraph, file_path, file_name):
    answered_pairs = []
    tasks = []
    semaphore = asyncio.Semaphore(10)  # Limit to 10 concurrent tasks

    async def limited_asearch(qa):
        async with semaphore:
            question = [qa_pair["question"] for qa_pair in golden if qa_pair["id"] == qa["id"]][0]
            time.sleep(0.01 * random.randint(1, 9))
            return await search_engine.asearch_with_context(question, qa["context"])

    for qa in graphgraph:
        tasks.append(limited_asearch(qa))

    results = await tqdm_asyncio.gather(*tasks, desc=f"Processing {file_name}")

    for qa, result in zip(graphgraph, results):
        answered_pairs.append({"id": qa["id"], "answer": result.response})

    with open(f"{file_path}/answer-{file_name}", "w") as fw:
        json.dump(answered_pairs, fw, indent=4)


async def main():
    api_key = os.environ["GRAPHRAG_API_KEY"]
    llm_model = "gpt-4o-mini"  # os.environ["GRAPHRAG_LLM_MODEL"]
    embedding_model = "text-embedding-3-small"  # os.environ["GRAPHRAG_EMBEDDING_MODEL"]

    llm = ChatOpenAI(
        api_key=api_key,
        model=llm_model,
        api_type=OpenaiApiType.OpenAI,  # OpenaiApiType.OpenAI or OpenaiApiType.AzureOpenAI
        max_retries=20,
    )

    token_encoder = tiktoken.get_encoding("cl100k_base")

    local_context_params = {
        "text_unit_prop": 0.5,
        "community_prop": 0.1,
        "conversation_history_max_turns": 5,
        "conversation_history_user_turns_only": True,
        "top_k_mapped_entities": 10,
        "top_k_relationships": 10,
        "include_entity_rank": True,
        "include_relationship_weight": True,
        "include_community_rank": False,
        "return_candidate_context": False,
        "embedding_vectorstore_key": EntityVectorStoreKey.ID,
        # set this to EntityVectorStoreKey.TITLE if the vectorstore uses entity title as ids
        "max_tokens": 12_000,
        # change this based on the token limit you have on your model (if you are using a model with 8k limit, a good setting could be 5000)
    }

    llm_params = {
        "max_tokens": 12_000,
        # change this based on the token limit you have on your model (if you are using a model with 8k limit, a good setting could be 1000=1500)
        "temperature": 0.0,
    }

    context_builder = LocalSearchMixedContext(
        community_reports=None,
        text_units=None,
        entities=[],
        relationships=None,
        # if you did not run covariates during indexing, set this to None
        covariates=None,  # covariates,
        entity_text_embeddings=None,
        embedding_vectorstore_key=EntityVectorStoreKey.ID,
        # if the vectorstore uses entity title as ids, set this to EntityVectorStoreKey.TITLE
        text_embedder=None,
        token_encoder=token_encoder,
    )

    search_engine = MyLocalSearch(
        llm=llm,
        context_builder=context_builder,
        token_encoder=token_encoder,
        llm_params=llm_params,
        context_builder_params=local_context_params
        # free form text describing the response type and format, can be anything, e.g. prioritized list, single paragraph, multiple paragraphs, multiple-page report
    )

    raw_golden_file = open(f"{ROOT_PATH}/qa-pairs/qa-pairs.json", 'r', encoding='utf-8')
    golden = json.load(raw_golden_file)

    # Find matching files
    directory_path = f'{ROOT_PATH}/output/'
    json_files = glob.glob(os.path.join(directory_path, 'ragtest-*.json'))

    print("#all files = ", len(json_files))
    semaphore = asyncio.Semaphore(4)  # Limit to 5 concurrent tasks

    async def limited_process(json_file):
        async with semaphore:
            with open(json_file, 'r', encoding='utf-8') as f:
                file_path, file_name = os.path.dirname(json_file).rstrip("/"), os.path.basename(json_file)
                graphgraph = json.load(f)
                await process_qa_pairs(search_engine, golden, graphgraph, file_path, file_name)

    tasks = [limited_process(json_file) for json_file in json_files]
    await tqdm_asyncio.gather(*tasks, desc="Processing all files")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("argument: shuf.py [root_path]")
        exit()
    ROOT_PATH = sys.argv[1]
    asyncio.run(main())


