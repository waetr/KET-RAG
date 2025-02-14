import logging
import time
import re
import string
from collections import Counter
from collections.abc import AsyncGenerator
from typing import Any

import tiktoken

from graphrag.model import (
    Community,
    CommunityReport,
    Covariate,
    Document,
    Entity,
    Relationship,
    TextUnit,
)
from graphrag.query.context_builder.builders import LocalContextBuilder
from graphrag.query.context_builder.conversation_history import (
    ConversationHistory,
)
from graphrag.query.llm.base import BaseLLM, BaseLLMCallback
from graphrag.query.llm.text_utils import num_tokens
from graphrag.query.structured_search.base import BaseSearch, SearchResult
from graphrag.query.structured_search.local_search.system_prompt import (
    LOCAL_SEARCH_SYSTEM_PROMPT,
)
from graphrag.query.structured_search.local_search.search import LocalSearch

LOCAL_SEARCH_EXACT_SYSTEM_PROMPT = """
---Role---

You are a helpful assistant responding to questions about data in the tables and supplementary materials provided.


---Goal---

Answer the user's question directly by extracting correct information from the data tables provided. The answer will be either a word, a phrase or a short sentence; the answer is supposed to be as short as possible.

If the answer can not be inferred from the data provided, say "Insufficient information." Do not make anything up.

For example, suppose the question is: "What country does the political movement started at the Christmas Meeting of 1888 seek sovereignty from?", your answer should be: "Denmark".

Do not include information where the supporting evidence for it is not provided in the data tables.

---Data tables---

{context_data}

---Goal---

Answer the user's question directly by extracting correct information from the data tables provided. The answer will be either a word, a phrase or a short sentence; the answer is supposed to be as short as possible.

If the answer can not be inferred from the data provided, say "Insufficient information." Do not make anything up.

For example, suppose the question is: "What country does the political movement started at the Christmas Meeting of 1888 seek sovereignty from?", your answer should be: "Denmark".

Do not include information where the supporting evidence for it is not provided in the data tables.
"""

class MyLocalSearch(LocalSearch):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build_context(
        self,
        query: str,
        conversation_history: ConversationHistory | None = None,
        supplement: str = "",
        selected_entities: list[Entity] | None = None,
        **kwargs,
    ) -> str:
        """Adopted from asearch"""
        """Build local search context that fits a single context window and generate answer for the user query."""
        # Ensure context_builder_params also has the updated value

        context_result = self.context_builder.build_context(
            query=query,
            conversation_history=conversation_history,
            predefined_selected_entities=selected_entities,
            **kwargs,
            **self.context_builder_params,
        )
        context_text = context_result.context_chunks
        if isinstance(context_text, list):
            context_text = "".join(context_text)

        return context_text + supplement

    async def asearch_with_context(
        self,
        query: str,
        context: str,
        **kwargs,
    ) -> SearchResult:
        """Build local search context that fits a single context window and generate answer for the user query."""
        start_time = time.time()
        search_prompt = ""

        try:
            search_prompt = LOCAL_SEARCH_EXACT_SYSTEM_PROMPT.format(
                context_data=context
            )
            search_messages = [
                {"role": "system", "content": search_prompt},
                {"role": "user", "content": query},
            ]

            response = await self.llm.agenerate(
                messages=search_messages,
                streaming=True,
                callbacks=self.callbacks,
                **self.llm_params,
            )

            return SearchResult(
                response=response,
                context_data=context,
                context_text=context,
                completion_time=time.time() - start_time,
                llm_calls=1,
                prompt_tokens=num_tokens(search_prompt, self.token_encoder),
                output_tokens=num_tokens(response, self.token_encoder)
            )

        except Exception:
            return SearchResult(
                response="",
                context_data=context,
                context_text=context,
                completion_time=time.time() - start_time,
                llm_calls=1,
                prompt_tokens=num_tokens(search_prompt, self.token_encoder),
                output_tokens=0
            )

# Adapted from https://github.com/Alab-NII/2wikimultihop/blob/main/2wikimultihop_evaluate.py
def normalize_answer(s):

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def f1_score(prediction, ground_truth):
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)

    ZERO_METRIC = (0, 0, 0)

    if normalized_prediction in ['yes', 'no', "insufficient information"] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC
    if normalized_ground_truth in ['yes', 'no'] and normalized_ground_truth not in normalized_prediction:
        return ZERO_METRIC

    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return ZERO_METRIC
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1, precision, recall


def exact_match_score(prediction, ground_truth):
    return int(normalize_answer(prediction) == normalize_answer(ground_truth))