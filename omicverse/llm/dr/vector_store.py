"""Simple wrapper around ChromaDB using GPT4All embeddings.

This module exposes helper functions to create a vector store backed by
`GPT4AllEmbeddings` and perform basic CRUD operations on ChromaDB
collections.
"""

from __future__ import annotations

import uuid
from typing import Mapping, Sequence, Any

import chromadb
from chromadb.api.models.Collection import Collection
from chromadb.api.types import EmbeddingFunction
from chromadb.config import Settings
from langchain_community.embeddings import GPT4AllEmbeddings


class _GPT4AllEmbeddingFunction(EmbeddingFunction):
    """Adapter to use :class:`GPT4AllEmbeddings` with ChromaDB."""

    def __init__(self, model: str | None = None, **kwargs: Any) -> None:
        self._embeddings = GPT4AllEmbeddings(model=model, **kwargs)

    def __call__(self, texts: Sequence[str]) -> list[list[float]]:  # type: ignore[override]
        return self._embeddings.embed_documents(list(texts))


def create_store(
    name: str,
    *,
    persist_directory: str | None = None,
    model: str | None = None,
    **model_kwargs: Any,
) -> Collection:
    """Create or load a ChromaDB ``Collection`` using GPT4All embeddings.

    Parameters
    ----------
    name:
        Identifier of the collection.
    persist_directory:
        Optional path to persist database files. When ``None`` an in-memory
        database is used.
    model:
        Optional GPT4All embedding model name.
    **model_kwargs:
        Additional keyword arguments forwarded to ``GPT4AllEmbeddings``.
    """

    settings = Settings(persist_directory=persist_directory) if persist_directory else Settings()
    client = chromadb.Client(settings)
    embedding_fn = _GPT4AllEmbeddingFunction(model=model, **model_kwargs)
    return client.get_or_create_collection(name=name, embedding_function=embedding_fn)


def add_documents(
    collection: Collection,
    documents: Sequence[str],
    *,
    ids: Sequence[str] | None = None,
    metadatas: Sequence[Mapping[str, Any]] | None = None,
) -> Sequence[str]:
    """Add ``documents`` to ``collection``.

    Parameters
    ----------
    collection:
        Target ChromaDB collection.
    documents:
        Text passages to embed and store.
    ids:
        Optional identifiers for the documents. Generated when omitted.
    metadatas:
        Optional metadata dicts corresponding to each document.

    Returns
    -------
    Sequence[str]
        The identifiers of the inserted documents.
    """

    if ids is None:
        ids = [str(uuid.uuid4()) for _ in documents]

    collection.add(
        documents=list(documents),
        ids=list(ids),
        metadatas=list(metadatas) if metadatas is not None else None,
    )
    return ids


def query_store(
    collection: Collection,
    query: str,
    *,
    n_results: int = 3,
) -> Mapping[str, Any]:
    """Query ``collection`` for items similar to ``query``.

    Parameters
    ----------
    collection:
        Target ChromaDB collection.
    query:
        Natural language query string.
    n_results:
        Number of matches to return.
    """

    return collection.query(query_texts=[query], n_results=n_results)


def delete_documents(
    collection: Collection,
    *,
    ids: Sequence[str] | None = None,
    where: Mapping[str, Any] | None = None,
) -> None:
    """Remove documents from ``collection``.

    Parameters
    ----------
    collection:
        Target ChromaDB collection.
    ids:
        Specific document identifiers to delete.
    where:
        Metadata filter specifying which documents to remove.
    """

    collection.delete(ids=list(ids) if ids is not None else None, where=where)
