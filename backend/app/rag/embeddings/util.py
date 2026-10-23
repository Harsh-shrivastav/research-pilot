"""Helpers shared by the hosted embedding providers."""
from __future__ import annotations

import math


def resize_embedding(vector: list[float], target_dim: int) -> list[float]:
    """Project a provider's native-dimensionality embedding onto the app's
    fixed `EMBEDDING_DIM` storage width (truncate + renormalise, or
    zero-pad), so the same `document_chunks.embedding` column works no
    matter which hosted model produced the vector.

    Truncating a normalised embedding and renormalising keeps cosine
    similarity comparisons well-behaved; it does lose some precision
    relative to the model's native dimensionality, which is the trade-off
    for keeping one static schema across providers.
    """
    if len(vector) == target_dim:
        return vector
    if len(vector) > target_dim:
        resized = vector[:target_dim]
    else:
        resized = vector + [0.0] * (target_dim - len(vector))
    norm = math.sqrt(sum(v * v for v in resized))
    if norm > 0:
        resized = [v / norm for v in resized]
    return resized
