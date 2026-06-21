# cleaner.py
"""
este script se encarga de limpiar y transformar los datos con Spark.
Pasos:
1. Selecciona y renombra las columnas necesarias del CSV de hoteles
2. Elimina filas con texto vacío y duplicados
3. Filtra ratings inválidos (solo acepta valores entre 1 y 5)
4. Hace un inner join con el dataset de estados usando province = State Code
5. Crea la columna review_full_text juntando el título y el texto de la reseña
6. Crea la columna sentiment: 1 si el rating es 4 o más (positivo), 0 si no (negativo)
"""
#Imports
from pyspark.sql.functions import col, concat, lit, when

def mrmusculo(hotels_df, states_df):
    """
    Limpia y transforma los DataFrames de reseñas y estados.
    """
    # Paso 1: Selecciona y renombra las columnas necesarias del CSV de hoteles
    hotels_df = hotels_df.select(
        col("name").alias("hotel_name"),
        col("city"),
        col("province"),
        col("`reviews.date`").alias("review_date"),
        col("`reviews.rating`").alias("rating"),
        col("`reviews.text`").alias("review_text"),
        col("`reviews.title`").alias("review_title"),
        col("`reviews.username`").alias("username")
    )

    # Paso 2: Elimina filas con texto vacío y duplicados
    hotels_df = hotels_df.filter(col("review_text").isNotNull())
    hotels_df = hotels_df.dropDuplicates()

    # Paso 3: Filtra ratings inválidos
    hotels_df = hotels_df.filter((col("rating") >= 1) & (col("rating") <= 5))

    # Paso 4: Hace un inner join con el dataset de estados
    df = hotels_df.join(states_df, hotels_df["province"] == states_df["State Code"], how="inner")
    # Paso 5: Crea la columna review_full_text
    df = df.withColumn("review_full_text", concat(col("review_title"), lit(" "), col("review_text")))

    # Paso 6: Crea la columna sentiment
    df = df.withColumn("sentiment", when(col("rating") >= 4, 1).otherwise(0))

    # Paso 7: Conserva region y division del dataset de estados (enriquecimiento del join)
    return df.select(
        "hotel_name", "city", "province", "review_date", "rating",
        "review_full_text", "sentiment",
        col("Region").alias("region"),
        col("Division").alias("division"),
    )


if __name__ == "__main__":
    from loader import load_data
    hotels_df, states_df = load_data("data/raw/Datafiniti_Hotel_Reviews.csv", "data/raw/states.csv")
    df = mrmusculo(hotels_df, states_df)
    print(f"Registros después de limpiar: {df.count()}")