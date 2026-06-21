"""
Pipeline de transformación de variables para clasificación de reseñas hoteleras.

Entradas:
- review_full_text
- region
- division

Transformaciones:
- Tokenizer
- StopWordsRemover
- HashingTF
- IDF
- OneHotEncoder

Salida:
- features: será utilizada por Logistic Regression y Random Forest.
"""

from pyspark.ml import Pipeline
from pyspark.ml.feature import (
    Tokenizer,
    StopWordsRemover,
    HashingTF,
    IDF,
    StringIndexer,
    OneHotEncoder,
    VectorAssembler
)


def build_feature_pipeline(use_region_features=True):
    stages = []

    tokenizer = Tokenizer(
        inputCol="review_full_text",
        outputCol="words"
    )

    stopwords = StopWordsRemover(
        inputCol="words",
        outputCol="filtered_words"
    )

    hashing_tf = HashingTF(
        inputCol="filtered_words",
        outputCol="raw_features",
        numFeatures=2**15
    )

    idf = IDF(
        inputCol="raw_features",
        outputCol="tfidf_features"
    )

    stages += [tokenizer, stopwords, hashing_tf, idf]

    feature_cols = ["tfidf_features"]

    if use_region_features:
        region_indexer = StringIndexer(
            inputCol="region",
            outputCol="region_index",
            handleInvalid="keep"
        )

        division_indexer = StringIndexer(
            inputCol="division",
            outputCol="division_index",
            handleInvalid="keep"
        )

        encoder = OneHotEncoder(
            inputCols=["region_index", "division_index"],
            outputCols=["region_ohe", "division_ohe"]
        )

        stages += [region_indexer, division_indexer, encoder]
        feature_cols += ["region_ohe", "division_ohe"]

    assembler = VectorAssembler(
        inputCols=feature_cols,
        outputCol="features"
    )

    stages.append(assembler)

    return Pipeline(stages=stages)