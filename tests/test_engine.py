"""Test suite for the engine module."""

import pytest
import numpy as np
import cv2
from face_embeddings_lib import EmbeddingEngine


@pytest.fixture
def engine():
    """Fixture that initializes the embedding engine."""
    return EmbeddingEngine()


def test_engine_init(engine):
    """Test embedding engine initialization."""
    assert engine is not None
    assert hasattr(engine, "generate_face_embeddings")


def test_empty_bytes_returns_no_faces(engine):
    """Test that empty byte input returns 0 faces."""
    result = engine.generate_face_embeddings(b"")
    assert isinstance(result, dict)
    assert "face_count" not in result or result["face_count"] == 0

    # Mock YOLO and Facenet models to avoid large downloads


def test_output_field_types(engine, monkeypatch):
    """Test that the output structure and types are correct."""

    class DummyModel:
        """Test class model"""

        def __call__(self, img, verbose=False):
            class DummyBoxes:
                """Test class boxes"""

                xyxy = type(
                    "obj",
                    (),
                    {"cpu": lambda self: self, "numpy": lambda self: [[0, 0, 1, 1]]},
                )()

            class DummyResult:
                """Test result class"""

                boxes = DummyBoxes()

            return [DummyResult()]

    class DummyEmbeddingModel:
        """Test class EmbeddingModel"""

        def __call__(self, x):
            return type(
                "obj",
                (),
                {
                    "detach": lambda self: type(
                        "obj", (), {"numpy": lambda self: [[0.0] * 512]}
                    )()
                },
            )()

    monkeypatch.setattr(engine, "model", DummyModel())
    monkeypatch.setattr(engine, "embedding_model", DummyEmbeddingModel())

    img = np.zeros((100, 100, 3), dtype="uint8")
    _, img_bytes = cv2.imencode(".jpg", img)
    result = engine.generate_face_embeddings(img_bytes.tobytes())

    assert isinstance(result, dict)
    assert "face_embedding" in result
    assert isinstance(result["face_embedding"], list)
    assert "face_count" in result
