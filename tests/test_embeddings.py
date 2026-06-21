# test_embeddings.py
# Autor: Andres
#
# Pruebas unitarias del modulo embeddings (funcion add_embeddings): que agregue
# la columna 'embedding' con vectores de 384 dimensiones y que maneje el texto
# nulo (los 285 casos) sin fallar.
#
# NOTA: estas pruebas usan el modelo real (sentence-transformers). Si la libreria
# no esta instalada (por ejemplo fuera del contenedor), pytest las SALTA en vez
# de fallar, gracias a importorskip.

import pytest

# Si no esta sentence-transformers, se salta todo este archivo.
pytest.importorskip("sentence_transformers")

from etl import embeddings  # noqa: E402  (import despues de importorskip a proposito)

EMBEDDING_DIM = 384


def test_agrega_columna_embedding_de_384(spark):
    """add_embeddings crea la columna 'embedding' con vectores de 384 dims."""
    # Arrange: dos resenas con texto
    df = spark.createDataFrame(
        [("Hotel A", "great clean room"), ("Hotel B", "very dirty and noisy")],
        ["hotel_name", "review_full_text"],
    )
    # Act
    out = embeddings.add_embeddings(df).collect()
    # Assert
    assert len(out) == 2
    for fila in out:
        assert fila["embedding"] is not None
        assert len(fila["embedding"]) == EMBEDDING_DIM


def test_maneja_texto_nulo(spark):
    """Una resena con review_full_text nulo igual recibe un vector valido."""
    # Arrange: texto valido + texto NULO (se debe resolver con coalesce a "")
    df = spark.createDataFrame(
        [("Hotel A", "nice pool"), ("Hotel B", None)],
        ["hotel_name", "review_full_text"],
    )
    # Act
    out = embeddings.add_embeddings(df).collect()
    # Assert: ningun embedding es nulo, y todos tienen 384 dims
    assert len(out) == 2
    for fila in out:
        assert fila["embedding"] is not None
        assert len(fila["embedding"]) == EMBEDDING_DIM
