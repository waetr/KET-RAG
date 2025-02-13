from typing import Any
import pandas as pd
import numpy as np
from graphrag.index.cache import InMemoryCache
from graphrag.index.operations.embed_text import embed_text
from nltk.corpus import stopwords
from graphrag.index.operations.snapshot import snapshot
from graphrag.index.storage import PipelineStorage
from datashaper import VerbCallbacks, DelegatingVerbCallbacks
from nltk.tokenize import sent_tokenize, word_tokenize
from collections import defaultdict
import string
import json

async def build_keyword_index(
    split_text_units_df: pd.DataFrame,
    storage: PipelineStorage,
    callbacks: VerbCallbacks,
    embedding_strategy: dict[str, Any] | None = None,
):
    stop_words = set(stopwords.words('english'))
    sentence_records = []  # To store sentences and their corresponding chunk IDs
    word_locations = defaultdict(list)  # To merge word records by word

    for _, row in split_text_units_df.iterrows():
        chunk_id = row['id']
        chunk_text = row['chunk']
        # Tokenize the chunk into sentences
        sentences = sent_tokenize(chunk_text)
        translator = str.maketrans('', '', string.punctuation)

        for sentence_id, sentence in enumerate(sentences):
            # Split long sentences into parts of <= 8000 tokens
            words = word_tokenize(sentence)
            # words = sentence.translate(translator).lower().split()
            sentence_parts = [
                words[i:i + 8000] for i in range(0, len(words), 8000)
            ]

            for part_id, part in enumerate(sentence_parts):
                # Record the sentence part with its chunk and sentence IDs
                sentence_records.append({
                    'chunk_id': chunk_id,
                    'sentence_id': f"{chunk_id}_{sentence_id}_{part_id}",
                    'sentence': ' '.join(part)
                })
                # Extract and record words, filtering out stopwords
                filtered_words = [
                    word for word in part
                    if word not in stop_words and word not in string.punctuation
                ]
                for word in filtered_words:
                    word_record = {
                        'chunk_id': chunk_id,
                        'sentence_id': f"{chunk_id}_{sentence_id}_{part_id}",
                    }
                    word_locations[word].append(word_record)

    # Create DataFrames for sentences and words
    sentences_df = pd.DataFrame(sentence_records)
    
    sentences_df["sentence_embedding"] = await embed_text(
        sentences_df,
        callbacks=callbacks,
        cache=InMemoryCache(),
        embed_column="sentence",
        embedding_name="sentence_embedding",
        strategy=embedding_strategy
    )

    sentence_embeddings = {
        row['sentence_id']: row['sentence_embedding']
        for _, row in sentences_df.iterrows()
    }

    words_df = pd.DataFrame([
        {
            'word': word,
            'embedding':  np.mean(
                [
                    np.asarray(sentence_embeddings[word_record['sentence_id']])
                    for word_record in word_records if word_record['sentence_id'] in sentence_embeddings
                ],
                axis=0
            ).tolist(),
            'chunk_ids': list(set([word_record['chunk_id'] for word_record in word_records]))
        }
        for word, word_records in word_locations.items()
    ])

    words_df = words_df[words_df['embedding'].notna()]

    await snapshot(
        words_df,
        name="keyword_index",
        storage=storage,
        formats=["parquet"],
    )

