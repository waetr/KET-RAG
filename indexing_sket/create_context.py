import asyncio
import math
import copy
import os
import sys
import json
from typing import Any
import uuid
import re

import faiss
import numpy as np

import pandas as pd
import tiktoken

from collections import Counter, defaultdict
import string
from pathlib import Path
from graphrag.api.query import _get_embedding_store
from graphrag.query.context_builder.entity_extraction import (
    EntityVectorStoreKey,
)
from graphrag.model import (
    Community,
    CommunityReport,
    Covariate,
    Document,
    Entity,
    Relationship,
    TextUnit,
)
from graphrag.query.indexer_adapters import (
    read_indexer_covariates,
    read_indexer_entities,
    read_indexer_relationships,
    read_indexer_reports,
    read_indexer_text_units,
)
from graphrag.query.input.loaders.dfs import (
    store_entity_semantic_embeddings,
)
from graphrag.query.llm.text_utils import num_tokens
from graphrag.query.llm.oai.chat_openai import ChatOpenAI
from graphrag.query.llm.oai.embedding import OpenAIEmbedding
from graphrag.query.llm.oai.typing import OpenaiApiType
from graphrag.query.question_gen.local_gen import LocalQuestionGen
from graphrag.query.structured_search.local_search.mixed_context import (
    LocalSearchMixedContext,
)
from graphrag.query.structured_search.local_search.search import LocalSearch
from graphrag.vector_stores.lancedb import LanceDBVectorStore
from tqdm import tqdm
from tqdm.asyncio import tqdm as atqdm
from graphrag.query.input.loaders.dfs import (
    read_text_units
)

import random

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

import networkx as nx
from numpy import argsort

COMMUNITY_REPORT_TABLE = "create_final_community_reports"
ENTITY_TABLE = "create_final_nodes"
ENTITY_EMBEDDING_TABLE = "create_final_entities"
RELATIONSHIP_TABLE = "create_final_relationships"
COVARIATE_TABLE = "create_final_covariates"
TEXT_UNIT_TABLE = "create_final_text_units"
COMMUNITY_LEVEL = 2


def main():
    """ Pre-process the data."""
    if len(sys.argv) != 4:
        print("argument: create_context.py [root_path] [strategy_build_context] [budget]")
        exit()
    ROOT_PATH, strategy_build_context, budget= sys.argv[1], sys.argv[2], float(sys.argv[3])
    length_text_context = int(12_000 * (1.0 - budget))

    ROOT_PATH = ROOT_PATH.rstrip("/")
    OUTPUT_DIR = f"{ROOT_PATH}/output"
    LANCEDB_URI = f"{ROOT_PATH}/lancedb-new"

    # download NTLK tools for tokenization
    # nltk.download('punkt', quiet=True)
    # nltk.download('stopwords', quiet=True)
    # nltk.download('punkt_tab', quiet=True)

    # read nodes table to get community and degree data
    node_df = pd.read_parquet(f"{OUTPUT_DIR}/{ENTITY_TABLE}.parquet")
    entity_df = pd.read_parquet(f"{OUTPUT_DIR}/{ENTITY_EMBEDDING_TABLE}.parquet")
    entity_embedding_df = pd.read_parquet(f"{OUTPUT_DIR}/embeddings.entity.description.parquet").rename(columns={'embedding': 'description_embedding'})
    if 'description_embedding' in entity_df.columns:
        entity_df = entity_df.drop(columns=['description_embedding'])
    entity_df = entity_df.merge(entity_embedding_df, on='id', how='inner')
    entities = read_indexer_entities(node_df, entity_df, COMMUNITY_LEVEL)
    # load description embeddings to an in-memory lancedb vectorstore
    # to connect to a remote db, specify url and port values.
    description_embedding_store = LanceDBVectorStore(collection_name=f"entity.description")
    description_embedding_store.connect(db_uri=LANCEDB_URI)
    entity_description_embeddings = store_entity_semantic_embeddings(entities=entities, vectorstore=description_embedding_store)
    relationship_df = pd.read_parquet(f"{OUTPUT_DIR}/{RELATIONSHIP_TABLE}.parquet")
    relationships = read_indexer_relationships(relationship_df)
    report_df = pd.read_parquet(f"{OUTPUT_DIR}/{COMMUNITY_REPORT_TABLE}.parquet")
    reports = read_indexer_reports(report_df, node_df, COMMUNITY_LEVEL)
    text_unit_df = pd.read_parquet(f"{OUTPUT_DIR}/{TEXT_UNIT_TABLE}.parquet")
    text_unit_embedding_df = pd.read_parquet(f"{OUTPUT_DIR}/embeddings.text_unit.text.parquet")
    text_units = read_indexer_text_units(text_unit_df)
    
    # Merge embeddings into text_units
    embedding_dict = dict(zip(text_unit_embedding_df['id'], text_unit_embedding_df['embedding']))
    for text_unit in text_units:
        text_unit.text_embedding = embedding_dict[text_unit.id]
    
    token_encoder = tiktoken.get_encoding("cl100k_base")

    text_units, entities, relationships = split_chunks_and_others(text_units, entities, relationships, token_encoder, new_size=300)

    split_text_units_df = pd.read_parquet(f"{OUTPUT_DIR}/split_text_units.parquet")
    keyword_df = pd.read_parquet(f"{OUTPUT_DIR}/keyword_index.parquet")
    split_text_units = read_text_units(
        split_text_units_df,
        text_col='chunk',
        short_id_col=None,
        covariates_col=None,
        entities_col=None,
        relationships_col=None,
        embedding_col='text_embedding'
    )

    api_key = os.environ["GRAPHRAG_API_KEY"]

    llm_model = "gpt-4o-mini"
    embedding_model = "text-embedding-3-small"

    llm = ChatOpenAI(
        api_key=api_key,
        model=llm_model,
        api_type=OpenaiApiType.OpenAI,  # OpenaiApiType.OpenAI or OpenaiApiType.AzureOpenAI
        max_retries=20,
    )

    text_embedder = OpenAIEmbedding(
        api_key=api_key,
        api_base=None,
        api_type=OpenaiApiType.OpenAI,
        model=embedding_model,
        deployment_name=embedding_model,
        max_retries=20,
    )

    context_builder = LocalSearchMixedContext(
        community_reports=reports,
        text_units=text_units,
        entities=entities,
        relationships=relationships,
        # if you did not run covariates during indexing, set this to None
        covariates=None,
        entity_text_embeddings=description_embedding_store,
        embedding_vectorstore_key=EntityVectorStoreKey.ID,
        # if the vectorstore uses entity title as ids, set this to EntityVectorStoreKey.TITLE
        text_embedder=text_embedder,
        token_encoder=token_encoder,
    )

    local_context_params = {
        "text_unit_prop": 0.5,
        "community_prop": 0.0,
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
        "max_tokens": 12_000 - length_text_context
        # change this based on the token limit you have on your model (if you are using a model with 8k limit, a good setting could be 5000)
    }

    llm_params = {
        "max_tokens": 2_000,
        # change this based on the token limit you have on your model (if you are using a model with 8k limit, a good setting could be 1000=1500)
        "temperature": 0.0,
    }

    search_engine = LocalSearch(
        llm=llm,
        context_builder=context_builder,
        token_encoder=token_encoder,
        llm_params=llm_params,
        context_builder_params=local_context_params,
        response_type="A single phrase",
    )

    # Preprocess the index
    candidate_units_dict = {unit.id: unit for unit in split_text_units}
    if strategy_build_context == "keyword":
        word_embeddings = {row['word']: row['embedding'] for _, row in keyword_df.iterrows()}
        word_chunks = {row['word']: row['chunk_ids'] for _, row in keyword_df.iterrows()}
        words = list(word_embeddings.keys())
        chunk_vecs = np.array([word_embeddings[w] for w in words], dtype=np.float32)

        index = faiss.IndexFlatIP(chunk_vecs.shape[1])
        index.add(chunk_vecs)

        token_counts = {unit.id: num_tokens(unit.text, token_encoder) for unit in split_text_units}
        # For each word, precompute chunk_ids and chunk_embeddings arrays
        word_chunk_data = {}
        for w in words:
            chunk_ids = list(word_chunks.get(w, []))
            chunk_embs = np.stack([candidate_units_dict[cid].text_embedding for cid in chunk_ids]).astype(np.float32)
            chunk_token_counts = [token_counts[cid] for cid in chunk_ids]
            word_chunk_data[w] = (chunk_ids, chunk_embs, chunk_token_counts)
        text_context_params = {
            'embedder': text_embedder,
            'encoder': token_encoder,
            'index': index,
            'words': words,
            'word_chunk_data': word_chunk_data,
            'candidate_units_dict': candidate_units_dict,
            'chunks_size': length_text_context
        }
    elif strategy_build_context == "text":
        chunk_ids = list(candidate_units_dict.keys())
        chunk_vecs = np.array([unit.text_embedding for unit in candidate_units_dict.values()], dtype=np.float32)
        index = faiss.IndexFlatIP(chunk_vecs.shape[1])
        index.add(chunk_vecs)
        text_context_params = {
            'embedder': text_embedder,
            'encoder': token_encoder,
            'index': index,
            'ids': chunk_ids,
            'candidate_units_dict': candidate_units_dict,
            'chunks_size': length_text_context
        }
    else:
        text_context_params = {}

    print("Start benchmark...")
    qa_file_path = f"{ROOT_PATH}/input/qa-pairs.json"
    folder_name = os.path.basename(os.path.normpath(f"{ROOT_PATH}"))
    output_path = f"{OUTPUT_DIR}/{folder_name}-{strategy_build_context}-{budget}.json"
    with open(qa_file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    # Filter out entries where question or answer is not a string.
    valid_pairs = [entry for entry in data if
                   isinstance(entry['question'], str) and isinstance(entry['answer'], str)][:5]

    max_concurrent_tasks = 1  # Adjust based on your system's capacity
    matched_pairs = asyncio.run(
        process_all_pairs(
            valid_pairs,
            strategy_build_context,
            budget,
            search_engine,
            text_context_params,
            max_concurrent_tasks
        )
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(matched_pairs, f, indent=4)
    # Output the match rate
    print(f'Stored in {output_path}')

def normalize_embeddings(vectors):
    if len(vectors.shape) == 1:
        return vectors / np.linalg.norm(vectors)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / norms

async def process_pair(pair, strategy_build_context, budget, search_engine, text_context_params, semaphore):
    async with semaphore:
        question = pair['question']
        context = await asyncio.to_thread(
            generate_text_unit_context,
            strategy_build_context,
            budget,
            search_engine,
            question,
            **text_context_params
        )
        return {"id": pair["id"], "context": context}

async def process_all_pairs(valid_pairs, strategy_build_context, budget, search_engine, text_context_params, max_concurrent_tasks):
    semaphore = asyncio.Semaphore(max_concurrent_tasks)
    tasks = [
        process_pair(pair, strategy_build_context, budget, search_engine, text_context_params, semaphore)
        for pair in valid_pairs
    ]
    return await atqdm.gather(*tasks, desc="Processing", unit="pair")


def generate_text_unit_context(
        strategy_build_context: str,
        budget: float,
        search_engine: LocalSearch,
        question: str,
        **kwargs
) -> str:
    if strategy_build_context == "keyword":
        text_content = find_chunks_keyword(question, **kwargs)
    elif strategy_build_context == "text":
        text_content = find_chunks_onehop(question, **kwargs)
    else:
        text_content = "\n\n-----Text source that may be relevant-----\n\nN/A\n"
    if budget > 0:
        context_params = {
            "text_unit_prop": 0.5,
            "community_prop": 0.0,
            "conversation_history_max_turns": 10,
            "conversation_history_user_turns_only": True,
            "top_k_mapped_entities": 10,
            "top_k_relationships": 10,
            "include_entity_rank": True,
            "include_relationship_weight": True,
            "include_community_rank": False,
            "return_candidate_context": False,
            # set this to EntityVectorStoreKey.TITLE if the vectorstore uses entity title as ids
            "max_tokens": 12000 - num_tokens(text_content, search_engine.context_builder.token_encoder)
            # change this based on the token limit you have on your model (if you are using a model with 8k limit, a good setting could be 5000)
        }
        context_result = search_engine.context_builder.build_context(question, **context_params)
        graph_context = context_result.context_chunks
    else:
        graph_context = ""
    return graph_context + text_content

def find_chunks_onehop(
        query: str,
        embedder: OpenAIEmbedding,
        encoder: tiktoken.Encoding,
        index: faiss.IndexFlatIP,  # pre-built FAISS index for words
        ids: list,  # list of words corresponding to the embeddings in 'index'
        candidate_units_dict: dict,
        chunks_size: int = 0
) -> str:
    if chunks_size == 0:
        return "\n\n-----Text source that may be relevant-----\n\nN/A\n"
    query_embedding = embedder.embed(query)
    query_embedding = np.array(query_embedding, dtype=np.float32).reshape(1, -1)
    # text_ids = [unit.id for unit in candidate_units_dict.values()]
    # text_embeddings = np.array([unit.text_embedding for unit in candidate_units_dict.values()], dtype=np.float32)
    # text_sims = list(cosine_similarity(query_embedding, text_embeddings).flatten())

    D, Idx = index.search(query_embedding, 1000)
    text_ranks = [(ids[i], float(D[0, idx])) for idx, i in enumerate(Idx[0])]

    #text_ranks = sorted(zip(text_ids, text_sims), key=lambda item: -item[1])

    text = "\n\n-----Text source that may be relevant-----\nid|text\n"
    text_tokens = num_tokens(text, encoder)
    for idx, (element_id, rank) in enumerate(text_ranks):
        element_content = candidate_units_dict[element_id].text
        newly_added_tokens = num_tokens(f"chunk_{idx+1}|" + element_content + "\n", encoder)
        if text_tokens + newly_added_tokens > chunks_size:
            break
        text = text + f"chunk_{idx+1}|" + element_content + "\n"
        text_tokens += newly_added_tokens
    return text

def find_chunks_keyword(
    query: str,
    embedder: OpenAIEmbedding,
    encoder: tiktoken.Encoding,
    # Precomputed data passed in as arguments instead of relying on external variables
    index: faiss.IndexFlatIP,    # pre-built FAISS index for words
    words: list,                 # list of words corresponding to the embeddings in 'index'
    word_chunk_data: dict,       # {word: (chunk_ids, chunk_embs, chunk_token_counts)}
    candidate_units_dict: dict,  # {unit_id: TextUnit}
    chunks_size: int,
) -> str:
    if chunks_size == 0:
        return "\n\n-----Text source that may be relevant-----\n\nN/A\n"
    query_embedding = embedder.embed(query)
    query_embedding = np.array(query_embedding, dtype=np.float32).reshape(1, -1)

    D, Idx = index.search(query_embedding, 1000)
    word_rank = [(words[i], float(D[0, idx])) for idx, i in enumerate(Idx[0])]

    selected_chunks = []
    tokens_original = 0
    visited_chunks = set()
    is_full = False
    for word, _ in word_rank:
        if is_full:
            break
        if word not in word_chunk_data:
            continue
        chunk_ids, chunk_embs, chunk_token_counts = word_chunk_data[word]

        # Compute similarities via dot product for normalized embeddings
        sims = (query_embedding @ chunk_embs.T).flatten()

        # Sort chunks by similarity (descending)
        idx_sorted = np.argsort(-sims)
        tokens_now = 0

        for idx_c in idx_sorted:
            chunk_id = chunk_ids[idx_c]
            sim = sims[idx_c]
            chunk_tokens = chunk_token_counts[idx_c]

            if tokens_original + chunk_tokens > chunks_size * 2:
                is_full = True
                break
            if chunk_id in visited_chunks:
                continue
            visited_chunks.add(chunk_id)
            tokens_original += chunk_tokens
            tokens_now += chunk_tokens
            selected_chunks.append((chunk_id, sim))
    selected_chunks.sort(key=lambda x: x[1], reverse=True)

    text = "\n\n-----Text source that may be relevant-----\nid|text\n"
    text_tokens = num_tokens(text, encoder)

    for idx, (element_id, sim) in enumerate(selected_chunks):
        new_text_segment = f"chunk_{idx+1}|" + candidate_units_dict[element_id].text + "\n"
        newly_added_tokens = num_tokens(new_text_segment, encoder)
        if text_tokens + newly_added_tokens > chunks_size:
            break
        text += new_text_segment
        text_tokens += newly_added_tokens

    return text

def split_chunks_and_others(chunks: list[TextUnit], entities: list[Entity], relations: list[Relationship], token_encoder, new_size=150
                            ) -> tuple[list[TextUnit], list[Entity], list[Relationship]]:
    results = []
    split_count = {}
    for chunk in chunks:
        text_token = token_encoder.encode(chunk.text)
        split_texts = [token_encoder.decode(text_token[i:i + new_size]) for i in range(0, len(text_token), new_size)]
        split_count[chunk.id] = len(split_texts)
        for idx, text in enumerate(split_texts):
            new_chunk = copy.deepcopy(chunk)
            new_chunk.text = text
            new_chunk.id = f"{new_chunk.id}_{idx}"
            results.append(new_chunk)
    for i in range(len(entities)):
        ids = entities[i].text_unit_ids
        new_ids = []
        for chunk_id in ids:
            new_ids.extend([f"{chunk_id}_{j}" for j in range(split_count.get(chunk_id, 0))])
        entities[i].text_unit_ids = new_ids
    for i in range(len(relations)):
        ids = relations[i].text_unit_ids
        new_ids = []
        for chunk_id in ids:
            new_ids.extend([f"{chunk_id}_{j}" for j in range(split_count.get(chunk_id, 0))])
        relations[i].text_unit_ids = new_ids

    return results, entities, relations


if __name__ == "__main__":
    main()

