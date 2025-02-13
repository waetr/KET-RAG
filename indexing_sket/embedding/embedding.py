# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""OpenAI Embedding model implementation."""

import asyncio
from collections.abc import Callable
from typing import Any

import numpy as np
import tiktoken
from tenacity import (
    AsyncRetrying,
    RetryError,
    Retrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from graphrag.logging import StatusLogger
from graphrag.query.llm.oai.base import OpenAILLMImpl
from graphrag.query.llm.oai.typing import (
    OPENAI_RETRY_ERROR_TYPES,
    OpenaiApiType,
)
from graphrag.query.llm.text_utils import chunk_text


class OpenAIBatchAsyncEmbedding(OpenAILLMImpl):
    """Wrapper for OpenAI Embedding models."""

    def __init__(
        self,
        api_key: str | None = None,
        azure_ad_token_provider: Callable | None = None,
        model: str = "text-embedding-3-small",
        deployment_name: str | None = None,
        api_base: str | None = None,
        api_version: str | None = None,
        api_type: OpenaiApiType = OpenaiApiType.OpenAI,
        organization: str | None = None,
        encoding_name: str = "cl100k_base",
        max_tokens: int = 8191,
        max_retries: int = 10,
        request_timeout: float = 180.0,
        retry_error_types: tuple[type[BaseException]] = OPENAI_RETRY_ERROR_TYPES,  # type: ignore
        reporter: StatusLogger | None = None,
    ):
        OpenAILLMImpl.__init__(
            self=self,
            api_key=api_key,
            azure_ad_token_provider=azure_ad_token_provider,
            deployment_name=deployment_name,
            api_base=api_base,
            api_version=api_version,
            api_type=api_type,  # type: ignore
            organization=organization,
            max_retries=max_retries,
            request_timeout=request_timeout,
            reporter=reporter,
        )

        self.model = model
        self.encoding_name = encoding_name
        self.max_tokens = max_tokens
        self.token_encoder = tiktoken.get_encoding(self.encoding_name)
        self.retry_error_types = retry_error_types

    async def aembed_with_retry1(self, text: list[str], **kwargs: Any) -> list[list[float]]:
        return [[0.0, 0.0],]

    async def aembed_with_retry(self, text: list[str], **kwargs: Any) -> list[list[float]]:
        """
        Attempt to embed text or a batch of texts with retries in case of errors.

        Args:
            text: A single string or a list of strings to embed.
            **kwargs: Additional parameters for the embedding request.

        Returns:
            A tuple containing the embedding(s) and the length of the input.
            For a single string, the length is the token count.
            For a list of strings, the length is the number of items in the list.
        """
        try:
            retryer = AsyncRetrying(
                stop=stop_after_attempt(self.max_retries),
                wait=wait_exponential_jitter(max=10),
                reraise=True,
                retry=retry_if_exception_type(self.retry_error_types),
            )
            async for attempt in retryer:
                with attempt:
                    embedding_response = await self.async_client.embeddings.create(
                        input=text,
                        model=self.model,
                        **kwargs,
                    )
                    embeddings = [item.embedding for item in embedding_response.data]
                    return embeddings
        except RetryError as e:
            self._reporter.error(
                message="Error at _aembed_with_retry()",
                details={self.__class__.__name__: str(e)},
            )
            return []
        else:
            return []

    async def aembed_with_retry2(self, text: list[str], **kwargs: Any) -> list[list[float]]:
        embedding_response = await self.async_client.embeddings.create(
            input=text,
            model=self.model,
            **kwargs,
        )
        embeddings = [item.embedding for item in embedding_response.data]
        return embeddings
