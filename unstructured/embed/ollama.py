from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Optional

import os
import numpy as np

from unstructured.documents.elements import (
    Element,
)
from unstructured.embed.interfaces import BaseEmbeddingEncoder, EmbeddingConfig
from unstructured.ingest.enhanced_dataclass import enhanced_field
from unstructured.ingest.error import EmbeddingEncoderConnectionError
from unstructured.utils import requires_dependencies

if TYPE_CHECKING:
    from langchain_community.embeddings import OllamaEmbeddings


@dataclass
class OllamaEmbeddingConfig(EmbeddingConfig):
    base_url: str = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
    model_name: str = "mxbai-embed-large"
    api_key: str = 'unused'

@dataclass
class OllamaEmbeddingEncoder(BaseEmbeddingEncoder):
    config: OllamaEmbeddingConfig
    _client: Optional["OllamaEmbeddings"] = field(init=False, default=None)
    _exemplary_embedding: Optional[List[float]] = field(init=False, default=None)

    @property
    def client(self) -> "OllamaEmbeddings":
        if self._client is None:
            self._client = self.create_client()
        return self._client

    @property
    def exemplary_embedding(self) -> List[float]:
        if self._exemplary_embedding is None:
            self._exemplary_embedding = self.client.embed_query("Q")
        return self._exemplary_embedding

    def initialize(self):
        pass

    def num_of_dimensions(self):
        return np.shape(self.exemplary_embedding)

    def is_unit_vector(self):
        return np.isclose(np.linalg.norm(self.exemplary_embedding), 1.0)

    def embed_query(self, query):
        return self.client.embed_query(str(query))

    def embed_documents(self, elements: List[Element]) -> List[Element]:
        embeddings = self.client.embed_documents([str(e) for e in elements])
        elements_with_embeddings = self._add_embeddings_to_elements(elements, embeddings)
        return elements_with_embeddings

    def _add_embeddings_to_elements(self, elements, embeddings) -> List[Element]:
        assert len(elements) == len(embeddings)
        elements_w_embedding = []
        for i, element in enumerate(elements):
            element.embeddings = embeddings[i]
            elements_w_embedding.append(element)
        return elements

    @EmbeddingEncoderConnectionError.wrap
    @requires_dependencies(
        ["langchain_community"],
        extras="openai",
    )
    def create_client(self) -> "OllamaEmbeddings":
        """Creates a langchain Ollama python client to embed elements."""
        from langchain_community.embeddings import OllamaEmbeddings

        openai_client = OllamaEmbeddings(
            base_url=self.config.base_url,
            model=self.config.model_name,  # type:ignore
        )
        return openai_client
