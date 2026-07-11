import math
import os
from typing import Any

import numpy as np
from fastapi import APIRouter, HTTPException
from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field


router = APIRouter()

EMBEDDING_MODEL = "text-embedding-3-small"
AIPIPE_BASE_URL = "https://aipipe.org/openai/v1"


class SemanticSearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query_id: str
    query: str
    candidates: list[str] = Field(min_length=3)


class SemanticSearchResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ranking: list[int]


def get_embedding_client() -> OpenAI:
    """
    Create an OpenAI-compatible client that sends requests through AI Pipe.
    """

    token = os.getenv("AIPIPE_TOKEN")

    if not token:
        raise HTTPException(
            status_code=500,
            detail="AIPIPE_TOKEN is not configured",
        )

    print(
        f"Q8 provider: {AIPIPE_BASE_URL}",
        flush=True,
    )

    return OpenAI(
        api_key=token,
        base_url=AIPIPE_BASE_URL,
        timeout=120.0,
        max_retries=2,
    )


def validate_input(
    query: str,
    candidates: list[str],
) -> tuple[str, list[str]]:
    cleaned_query = query.strip()

    if not cleaned_query:
        raise HTTPException(
            status_code=400,
            detail="query cannot be empty",
        )

    if len(candidates) < 3:
        raise HTTPException(
            status_code=400,
            detail="At least 3 candidates are required",
        )

    cleaned_candidates: list[str] = []

    for index, candidate in enumerate(candidates):
        cleaned_candidate = candidate.strip()

        if not cleaned_candidate:
            raise HTTPException(
                status_code=400,
                detail=f"candidates[{index}] cannot be empty",
            )

        cleaned_candidates.append(cleaned_candidate)

    return cleaned_query, cleaned_candidates


def create_embeddings(
    texts: list[str],
) -> np.ndarray:
    client = get_embedding_client()

    try:
        response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=texts,
            encoding_format="float",
        )

    except Exception as error:
        print(
            "Q8 embedding error: "
            f"{type(error).__name__}: {error}",
            flush=True,
        )

        raise HTTPException(
            status_code=502,
            detail=f"Embedding request failed: {error}",
        ) from error

    if not response.data:
        raise HTTPException(
            status_code=502,
            detail="Embedding API returned no vectors",
        )

    ordered_items = sorted(
        response.data,
        key=lambda item: item.index,
    )

    if len(ordered_items) != len(texts):
        raise HTTPException(
            status_code=502,
            detail=(
                "Embedding count mismatch: "
                f"expected {len(texts)}, "
                f"received {len(ordered_items)}"
            ),
        )

    try:
        vectors = np.asarray(
            [
                item.embedding
                for item in ordered_items
            ],
            dtype=np.float64,
        )
    except (TypeError, ValueError) as error:
        raise HTTPException(
            status_code=502,
            detail="Embedding API returned invalid vectors",
        ) from error

    if vectors.ndim != 2:
        raise HTTPException(
            status_code=502,
            detail="Embedding vectors have an invalid shape",
        )

    if not np.all(np.isfinite(vectors)):
        raise HTTPException(
            status_code=502,
            detail="Embedding vectors contain invalid numbers",
        )

    return vectors


def cosine_similarity_scores(
    query_vector: np.ndarray,
    candidate_vectors: np.ndarray,
) -> np.ndarray:
    query_norm = np.linalg.norm(query_vector)

    candidate_norms = np.linalg.norm(
        candidate_vectors,
        axis=1,
    )

    if (
        not math.isfinite(float(query_norm))
        or query_norm == 0
    ):
        raise HTTPException(
            status_code=502,
            detail="Query embedding has invalid magnitude",
        )

    if np.any(candidate_norms == 0):
        raise HTTPException(
            status_code=502,
            detail="A candidate embedding has zero magnitude",
        )

    scores = (
        candidate_vectors @ query_vector
    ) / (
        candidate_norms * query_norm
    )

    if not np.all(np.isfinite(scores)):
        raise HTTPException(
            status_code=502,
            detail="Cosine similarity produced invalid values",
        )

    return scores


def top_three_indices(
    scores: np.ndarray,
) -> list[int]:
    """
    Return the 3 highest-scoring candidate indices.

    Stable sorting ensures smaller index wins exact ties.
    """

    indices = np.argsort(
        -scores,
        kind="stable",
    )[:3]

    return [
        int(index)
        for index in indices.tolist()
    ]


@router.post(
    "/semantic-search",
    response_model=SemanticSearchResponse,
)
def semantic_search(
    request: SemanticSearchRequest,
) -> dict[str, Any]:
    query, candidates = validate_input(
        query=request.query,
        candidates=request.candidates,
    )

    print(
        f"Q8 request: query_id={request.query_id}, "
        f"candidates={len(candidates)}",
        flush=True,
    )

    all_texts = [
        query,
        *candidates,
    ]

    vectors = create_embeddings(all_texts)

    query_vector = vectors[0]
    candidate_vectors = vectors[1:]

    scores = cosine_similarity_scores(
        query_vector=query_vector,
        candidate_vectors=candidate_vectors,
    )

    ranking = top_three_indices(scores)

    print(
        f"Q8 ranking for {request.query_id}: {ranking}",
        flush=True,
    )

    print(
        "Q8 top scores: "
        + str(
            [
                {
                    "index": index,
                    "score": float(scores[index]),
                }
                for index in ranking
            ]
        ),
        flush=True,
    )

    return {
        "ranking": ranking,
    }


@router.get("/semantic-search")
def semantic_search_information():
    return {
        "message": (
            "Use POST with query_id, query and candidates"
        )
    }