# loader.py
"""
este script se encarga de leer los dos archivos CSV con Spark y devolverlos como DataFrames.
No modifica los datos, solo los carga.
Pasos:
1. Inicia una sesión de Spark
2. Lee el CSV de reseñas de hoteles
3. Lee el CSV de estados de EE.UU.
4. Devuelve los dos DataFrames
"""
#Imports
from pyspark.sql import SparkSession

def load_data(hotels_path, states_path):
    """
    Lee el CSV y devuelve dos DataFrames: uno con las reseñas de hoteles y otro con los estados.
    """
    # Paso 1: Inicia una sesión de Spark
    spark = SparkSession.builder.appName("HotelReviewsLoader").getOrCreate()
    # Paso 2: Lee el CSV de reseñas de hoteles
    hotels_df = spark.read.csv(hotels_path, header=True, inferSchema=True)
    # Paso 3: Lee el CSV de estados de EE.UU.
    states_df = spark.read.csv(states_path, header=True, inferSchema=True)
    # Paso 4: Devuelve los dos DataFrames
    return hotels_df, states_df

if __name__ == "__main__":
    hotels_path = "data/raw/Datafiniti_Hotel_Reviews.csv"
    states_path = "data/raw/states.csv"
    hotels_df, states_df = load_data(hotels_path, states_path)
    print(f"Hoteles: {hotels_df.count()} filas, {len(hotels_df.columns)} columnas")
    print(f"Estados: {states_df.count()} filas, {len(states_df.columns)} columnas")


