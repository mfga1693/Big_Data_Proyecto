"""
Programa principal para entrenar y comparar los modelos de predicción.

Aquí se orquesta el flujo completo de entrenamiento y evaluación de los modelos de clasificación:
Se lee el parquet generado por el ETL, se seleccionan las columnas necesarias, se divide el dataset en entrenamiento
y prueba, se entrenan Logistic Regression y Random Forest, y finalmente se calculan métricas para comparar ambos modelos.

Modelos implementados:
- Logistic Regression
- Random Forest
"""

from pyspark.sql import SparkSession

from src.models.logistic_regression import train_logistic_regression
from src.models.random_forest import train_random_forest, get_feature_importance


def print_metrics(metrics):
    """
    Imprime las métricas de evaluación de un modelo.

    Entrada:
    - metrics: diccionario con accuracy, precision, recall, f1 y auc_roc.
    """

    print(f"\nModelo: {metrics['model']}")
    print(f"Accuracy : {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall   : {metrics['recall']:.4f}")
    print(f"F1-score : {metrics['f1']:.4f}")
    print(f"AUC-ROC  : {metrics['auc_roc']:.4f}")


def print_comparison(lr_metrics, rf_metrics):
    """
    Imprime una comparación  entre ambos modelos usando las mismas métricas, 
    con el fin de evaluar cuál tuvo mejor desempeño bajo las mismas condiciones..
    """

    print("\n==============================")
    print("Comparación de modelos")
    print("==============================")

    print(f"{'Métrica':<12}{'Logistic Regression':<25}{'Random Forest':<20}")
    print("-" * 57)

    for metric in ["accuracy", "precision", "recall", "f1", "auc_roc"]:
        print(
            f"{metric:<12}"
            f"{lr_metrics[metric]:<25.4f}"
            f"{rf_metrics[metric]:<20.4f}"
        )

    # Se usa F1-score como métrica principal para la conclusión preliminar,
    # ya que combina precision y recall, y es más informativa que accuracy
    # cuando existe cierto desbalance entre clases.
    if lr_metrics["f1"] > rf_metrics["f1"]:
        print("\nConclusión preliminar: Logistic Regression obtuvo mejor F1-score.")
    elif rf_metrics["f1"] > lr_metrics["f1"]:
        print("\nConclusión preliminar: Random Forest obtuvo mejor F1-score.")
    else:
        print("\nConclusión preliminar: ambos modelos obtuvieron el mismo F1-score.")


def show_confusion_matrix(predictions, model_name):
    """
    Muestra la matriz de confusión de un modelo.

    La matriz de confusión permite observar:
    - cuántas reseñas negativas fueron clasificadas correctamente como negativas.
    - cuántas reseñas positivas fueron clasificadas correctamente como positivas.
    - cuántos errores cometió el modelo en cada clase.

    Entradas:
    - predictions: DataFrame con las columnas sentiment y prediction.
    - model_name: nombre del modelo evaluado.
    """

    print(f"\nMatriz de confusión - {model_name}")
    (
        predictions
        .groupBy("sentiment", "prediction")
        .count()
        .orderBy("sentiment", "prediction")
        .show()
    )


def main():
    """
    Ejecuta el flujo completo de entrenamiento y evaluación de modelos.
    """

    # Se crea la sesión de Spark. Esta sesión permite leer el parquet
    # y ejecutar los modelos de Spark MLlib.
    spark = (
        SparkSession.builder
        .appName("HotelReviews-Models")
        .getOrCreate()
    )


    parquet_path = "data/processed/hotel_reviews_embeddings.parquet"

    print(f"Leyendo datos desde: {parquet_path}")
    df = spark.read.parquet(parquet_path)

    # Se imprime el esquema para verificar que el parquet contiene
    # las columnas necesarias antes de entrenar los modelos.
    print("\nSchema del DataFrame:")
    df.printSchema()

    print("\nColumnas disponibles:")
    print(df.columns)

    # Columnas mínimas requeridas para entrenar los modelos.
    required_columns = ["review_full_text", "sentiment", "region", "division"]

    # Validación: si falta alguna columna, se detiene la ejecución
    for column in required_columns:
        if column not in df.columns:
            raise ValueError(f"Falta la columna requerida: {column}")

    # Se seleccionan únicamente las columnas necesarias para el modelo.
    # La columna embedding se ignora porque corresponde a búsqueda semántica,
    # no al pipeline TF-IDF definido para los modelos predictivos.
    df_model = (
        df
        .select("review_full_text", "sentiment", "region", "division")
        .dropna(subset=["review_full_text", "sentiment", "region", "division"])
    )

    # División de los datos: 80% para entrenamiento y 20% para prueba.
    train_df, test_df = df_model.randomSplit([0.8, 0.2], seed=42)

    print(f"\nRegistros de entrenamiento: {train_df.count()}")
    print(f"Registros de prueba: {test_df.count()}")

    # Entrenamiento y evaluación del modelo Logistic Regression.
    print("\nEntrenando Logistic Regression...")
    lr_model, lr_predictions, lr_metrics = train_logistic_regression(
        train_df,
        test_df,
        use_region_features=True
    )

    print_metrics(lr_metrics)
    show_confusion_matrix(lr_predictions, "Logistic Regression")

    # Entrenamiento y evaluación del modelo Random Forest.
    print("\nEntrenando Random Forest...")
    rf_model, rf_predictions, rf_metrics = train_random_forest(
        train_df,
        test_df,
        use_region_features=True
    )

    print_metrics(rf_metrics)
    show_confusion_matrix(rf_predictions, "Random Forest")

    # Comparación final entre ambos modelos.
    print_comparison(lr_metrics, rf_metrics)

    # Feature importance del Random Forest: indica qué variables fueron más usadas por el bosque
    print("\nFeature importance de Random Forest:")
    feature_importance = get_feature_importance(rf_model)
    print(feature_importance)

    spark.stop()


if __name__ == "__main__":
    main()