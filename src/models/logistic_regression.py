"""

Modelo Logistic Regression: entrenamiento y evaluación del modelo.

Se utiliza logistic regression para la clasificación binaria del sentimiento
de las reseñas hoteleras, donde:
- 0: reseña negativa
- 1: reseña positiva

"""

from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import (
    BinaryClassificationEvaluator,
    MulticlassClassificationEvaluator
)

from src.models.pipeline import build_feature_pipeline


def train_logistic_regression(train_df, test_df, use_region_features=True):
    """
    Entrena y evalúa el modelo.
    
    Entradas:
    - train_df: Df de entrenamiento.
    - test_df: Df de prueba.
    - use_region_features: indica si se incluyen region y division como features.

    Salidas:
    - model: pipeline entrenado.
    - predictions: predicciones sobre el conjunto de prueba.
    - metrics: diccionario con métricas de evaluación.
    
    """

    feature_pipeline = build_feature_pipeline(
        use_region_features=use_region_features
    )

    lr = LogisticRegression(
        featuresCol="features",
        labelCol="sentiment",
        predictionCol="prediction",
        probabilityCol="probability",
        rawPredictionCol="rawPrediction",
        maxIter=100,
        regParam=0.01
    )

    full_pipeline = Pipeline(
        stages=feature_pipeline.getStages() + [lr]
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
        "model": "Logistic Regression",
        "accuracy": accuracy_evaluator.evaluate(predictions),
        "precision": precision_evaluator.evaluate(predictions),
        "recall": recall_evaluator.evaluate(predictions),
        "f1": f1_evaluator.evaluate(predictions),
        "auc_roc": auc_evaluator.evaluate(predictions)
    }

    return model, predictions, metrics