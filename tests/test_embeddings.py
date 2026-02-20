"""Tests for embedding generator."""

import numpy as np
import pytest
from code_similarity_mcp.embeddings.generator import EmbeddingGenerator


@pytest.fixture(scope="module")
def generator():
    return EmbeddingGenerator()


def test_encode_one_returns_correct_shape(generator):
    emb = generator.encode_one("func test(): pass")
    assert emb.shape == (384,)
    assert emb.dtype == np.float32


def test_encode_batch_returns_correct_shape(generator):
    texts = ["func a(): pass", "func b(): pass", "func c(): pass"]
    embs = generator.encode(texts)
    assert embs.shape == (3, 384)


def test_encode_empty_list(generator):
    embs = generator.encode([])
    assert embs.shape == (0, 384)


def test_embeddings_are_normalized(generator):
    emb = generator.encode_one("func test(): var x = 1")
    norm = np.linalg.norm(emb)
    assert abs(norm - 1.0) < 1e-5


def test_similar_code_has_high_cosine_similarity(generator):
    """Normalized code from equivalent functions should embed very closely."""
    from code_similarity_mcp.normalizer import normalize_code

    code_a = normalize_code("def calc(a, b):\n    return a + b")
    code_b = normalize_code("def compute(x, y):\n    return x + y")
    # After normalization both become identical -> cosine ~1.0
    emb_a = generator.encode_one(code_a)
    emb_b = generator.encode_one(code_b)
    cosine = float(np.dot(emb_a, emb_b))
    assert cosine > 0.99


def test_different_code_has_lower_similarity(generator):
    from code_similarity_mcp.normalizer import normalize_code

    code_a = normalize_code("func calc(a, b):\n    return a + b")
    code_b = normalize_code("func load_texture(path):\n    return ResourceLoader.load(path)")
    emb_a = generator.encode_one(code_a)
    emb_b = generator.encode_one(code_b)
    cosine = float(np.dot(emb_a, emb_b))
    assert cosine < 0.95
