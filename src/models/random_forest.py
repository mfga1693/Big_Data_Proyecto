"""
Modelo Random Forest: entrenamiento y evaluación del modelo.

Se utiliza Random Forest como segundo modelo de clasificación binaria
para comparar su desempeño contra Logistic Regression.
"""

from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import (
    BinaryClassificationEvaluator,
    MulticlassClassificationEvaluator
)

from src.models.pipeline import build_feature_pipeline


def train_random_forest(train_df, test_df, use_region_features=True):
    """
    Entrena y evalúa el modelo Random Forest.

    Entradas:
    - train_df: DataFrame de entrenamiento.
    - test_df: DataFrame de prueba.
    - use_region_features: indica si se incluyen region y division como features.

    Salidas:
    - model: pipeline entrenado.
    - predictions: predicciones sobre el conjunto de prueba.
    - metrics: diccionario con métricas de evaluación.
    """

    feature_pipeline = build_feature_pipeline(
        use_region_features=use_region_features
    )

    rf = RandomForestClassifier(
        featuresCol="features",
        labelCol="sentiment",
        predictionCol="prediction",
        probabilityCol="probability",
        rawPredictionCol="rawPrediction",
        numTrees=50,
        maxDepth=10,
        seed=42
    )

    full_pipeline = Pipeline(
        stages=feature_pipeline.getStages() + [rf]
    )

    model = full_pipeline.fit(train_df)

    predictions = model.transform(test_df)

    accuracy_evaluator = MulticlassClassificationEvaluator(
        labelCol="sentiment",
        predictionCol="prediction",
        metricName="accuracy"
    )

    precision_evaluator = MulticlassClassificationEvaluator(
        labelCol="sentiment",
        predictionCol="prediction",
        metricName="weightedPrecision"
    )

    recall_evaluator = MulticlassClassificationEvaluator(
        labelCol="sentiment",
        predictionCol="prediction",
        metricName="weightedRecall"
    )

    f1_evaluator = MulticlassClassificationEvaluator(
        labelCol="sentiment",
        predictionCol="prediction",
        metricName="f1"
    )

    auc_evaluator = BinaryClassificationEvaluator(
        labelCol="sentiment",
        rawPredictionCol="rawPrediction",
        metricName="areaUnderROC"
    )

    metrics = {
        "model": "Random Forest",
        "accuracy": accuracy_evaluator.evaluate(predictions),
        "precision": precision_evaluator.evaluate(predictions),
        "recall": recall_evaluator.evaluate(predictions),
        "f1": f1_evaluator.evaluate(predictions),
        "auc_roc": auc_evaluator.evaluate(predictions)
    }

    return model, predictions, metrics


def get_feature_importance(model):
    """
    Obtiene las importancias de variables del Random Forest.

    Entrada:
    - model: pipeline entrenado.

    Salida:
    - Vector de importancias.
    """

    rf_model = model.stages[-1]

    return rf_model.featureImportances