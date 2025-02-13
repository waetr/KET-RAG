import pandas as pd
import copy
import io
import random
import re
import numpy as np
import regex
from sklearn.metrics.pairwise import cosine_similarity
from typing import Any
from collections import Counter
from tqdm import tqdm

from graphrag.index.cache import InMemoryCache
from graphrag.query.input.loaders.dfs import (
    read_text_units
)
from graphrag.model import (
    TextUnit
)
from graphrag.index.utils import gen_md5_hash
from indexing_sket.embedding.embedding import OpenAIBatchAsyncEmbedding
from graphrag.query.llm.oai.embedding import OpenAIEmbedding
from graphrag.query.llm.oai.typing import OpenaiApiType
from graphrag.index.operations.embed_text import embed_text
import tiktoken
import networkx as nx
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from graphrag.index.operations.snapshot import snapshot
from graphrag.index.storage import PipelineStorage
from datashaper import VerbCallbacks
from indexing_sket.create_final_keyword_index import build_keyword_index

async def filter_text_units(
    text_units_df: pd.DataFrame,
    storage: PipelineStorage,
    callbacks: VerbCallbacks,
    chunk_strategy: dict[str, Any] | None = None,
    embedding_strategy: dict[str, Any] | None = None
) -> pd.DataFrame:
    budget = chunk_strategy['budget']
    build_skeleton_method = chunk_strategy['build_skeleton_method']
    knn_edges = chunk_strategy['knn_edges']
    split_size = chunk_strategy['split_size']
    random.seed(42)
    token_encoder = tiktoken.get_encoding(chunk_strategy.get('encoding_name', 'cl100k_base'))
    text_units = read_text_units(
        text_units_df,
        text_col='chunk',
        short_id_col=None,
        covariates_col=None,
        entities_col=None,
        relationships_col=None,
        embedding_col=None
    )
    text_units = split_chunks(text_units, token_encoder, split_size)
    text_embedding_bytes = await storage.get('split_text_units.parquet', as_bytes=True)
    if text_embedding_bytes is not None:
        print("Text Embedding file has been read!")
        buffer = io.BytesIO(text_embedding_bytes)
        text_embedding_df = pd.read_parquet(buffer)
    else:
        print("Generating Text Embedding file")
        text_embedding_df = pd.DataFrame([
            {
                "id": unit.id,
                "chunk": unit.text,
                "chunk_id": unit.id,
                "document_ids": unit.document_ids,
                "n_tokens": unit.n_tokens
            }
            for unit in text_units
        ])
        text_embedding_df["text_embedding"] = await embed_text(
            text_embedding_df,
            callbacks=callbacks,
            cache=InMemoryCache(),
            embed_column="chunk",
            embedding_name="text_embedding",
            strategy=embedding_strategy,
        )
        await snapshot(
            text_embedding_df,
            name="split_text_units",
            storage=storage,
            formats=["parquet"],
        )

    keyword_file = await storage.get('keyword_index.parquet', as_bytes=True)
    if keyword_file is None:
        print("Generating Keyword file")
        await build_keyword_index(text_embedding_df, storage, callbacks, embedding_strategy)
    else:
        print("Keyword file has been read!")

    text_units = read_text_units(
        text_embedding_df,
        text_col='chunk',
        short_id_col=None,
        covariates_col=None,
        entities_col=None,
        relationships_col=None,
        embedding_col='text_embedding'
    )
    sample_length = int(len(text_units) * budget)
    if build_skeleton_method == "uniform":
        random.shuffle(text_units)
        sampled_text_units = text_units[:sample_length]
    else:
        G = build_knn_chunk_graph(text_units, knn_edges, knn_edges)
        pr = nx.pagerank(G, alpha=0.85)
        sampled_text_units = sorted(text_units, key=lambda x: pr.get(x.id, 0), reverse=True)[:sample_length]

    sampled_text_units = sorted(sampled_text_units, key=lambda chunk: int(chunk.id))
    if split_size < chunk_strategy.get('chunk_size', 1200):
        sampled_text_units = merge_chunks(sampled_text_units, token_encoder, chunk_strategy.get('chunk_size', 1200))
    sampled_df = pd.DataFrame([
        {
            "id": (hash_value := gen_md5_hash({"chunk": unit.text}, ["chunk"])),
            "chunk": unit.text,
            "chunk_id": hash_value,
            "document_ids": unit.document_ids,
            "n_tokens": unit.n_tokens,
        }
        for unit in sampled_text_units
    ])

    return sampled_df


def split_chunks(chunks: list[TextUnit], token_encoder, new_size) -> list[TextUnit]:
    results = []
    total_num = 0
    for chunk in chunks:
        if new_size < 1200:
            text_token = token_encoder.encode(chunk.text)
            split_texts = [token_encoder.decode(text_token[i:i + new_size]) for i in range(0, len(text_token), new_size)]
        else:
            split_texts = [chunk.text]
        for text in split_texts:
            new_chunk = copy.deepcopy(chunk)
            new_chunk.text = text
            new_chunk.id = str(total_num)
            new_chunk.n_tokens = len(token_encoder.encode(text))
            results.append(new_chunk)
            total_num += 1
    return results

def merge_chunks(chunks: list[TextUnit], token_encoder, new_size) -> list[TextUnit]:
    results = []
    token_result, document_ids, current_id = [], set(), ""
    for chunk in chunks:
        text_tokens = token_encoder.encode(chunk.text)
        if len(token_result) + len(text_tokens) > new_size:
            new_chunk = TextUnit(id=current_id,
                                 short_id=current_id,
                                 text=token_encoder.decode(token_result),
                                 document_ids=list(document_ids),
                                 n_tokens=len(token_result))
            results.append(new_chunk)
            token_result, document_ids, current_id = [], set(), ""
        token_result += text_tokens
        document_ids.update(chunk.document_ids or [])
        current_id = current_id or chunk.id
    if token_result:
        new_chunk = TextUnit(id=current_id,
                             short_id=current_id,
                             text=token_encoder.decode(token_result),
                             document_ids=list(document_ids),
                             n_tokens=len(token_result))
        results.append(new_chunk)
    return results

def build_knn_chunk_graph(text_units: list[TextUnit], k1: int, k2: int) -> nx.Graph:
    # find all text-text pairs
    co_occurrence_text_pairs = find_top_k_cooccurrence_pairs(text_units, k=k1)
    similarity_text_pairs = find_top_k_similar_texts(text_units, k=k2)
    text_pairs = list(co_occurrence_text_pairs) + list(similarity_text_pairs)
    G = nx.Graph()
    for text_unit in text_units:
        G.add_node(text_unit.id)
    for u, v in text_pairs:
        G.add_edge(u, v)
    return G

def find_top_k_cooccurrence_pairs(text_units: list[TextUnit], k: int, bow_threshold: int = 2000) -> list[tuple[str, str]]:
    def remove_stopwords(text):
        stop_words = set(stopwords.words('english'))
        word_tokens = word_tokenize(text.lower())  # Tokenize and lowercase the text
        filtered_words = [word for word in word_tokens if word.isalpha() and word not in stop_words]
        word_freq = Counter(filtered_words)
        most_common_words = [word for word, freq in word_freq.most_common(bow_threshold)]
        return set(most_common_words)  # Return as a set of unique words

    # Step 1: Remove stop words from all text units
    processed_texts = {text_unit.id: remove_stopwords(text_unit.text) for text_unit in text_units}

    # Step 2: Calculate the co-occurrence of distinct words for each pair
    cooccurrence_pairs = {}
    keys = list(processed_texts.keys())

    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            words1 = processed_texts[keys[i]]
            words2 = processed_texts[keys[j]]
            # Find the number of co-occurring words
            common_words = words1.intersection(words2)
            cooccurrence_count = len(common_words)

            # Store the co-occurrence count in a dict with both orders (i->j and j->i)
            if keys[i] not in cooccurrence_pairs:
                cooccurrence_pairs[keys[i]] = []
            if keys[j] not in cooccurrence_pairs:
                cooccurrence_pairs[keys[j]] = []

            cooccurrence_pairs[keys[i]].append((cooccurrence_count, keys[j]))
            cooccurrence_pairs[keys[j]].append((cooccurrence_count, keys[i]))

    # Step 3: For each text unit, keep the top k pairs based on co-occurrence count
    top_k_pairs = []
    for key, pairs in cooccurrence_pairs.items():
        # Sort the pairs based on the co-occurrence count in descending order
        pairs_sorted = sorted(pairs, key=lambda x: x[0], reverse=True)

        # Keep only the top k pairs
        for count, other_key in pairs_sorted[:k]:
            top_k_pairs.append((key, other_key))

    return top_k_pairs


def find_top_k_similar_texts(text_units: list[TextUnit], k: int) -> list[tuple[str, str]]:
    text_keys = [text_unit.id for text_unit in text_units]
    embeddings = np.array([text.text_embedding for text in text_units])  # Assume embeddings are precomputed

    # Step 4: Compute pairwise cosine similarity
    cosine_similarities = cosine_similarity(embeddings)

    # Step 5: Find top-k most similar texts
    top_k_pairs = []
    for i, key in enumerate(text_keys):
        similarities = cosine_similarities[i]
        top_k_indices = np.argsort(-similarities)[1:k + 1]  # Exclude self similarity

        for idx in top_k_indices:
            similar_key = text_keys[idx]
            top_k_pairs.append((key, similar_key))

    return top_k_pairs
