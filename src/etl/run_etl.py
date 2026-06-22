# run_etl.py
"""
Programa principal del ETL: corre todo el flujo en UNA sola sesion de Spark.

    loader  ->  cleaner  ->  exporter (parquet limpio)  ->  embeddings (parquet con vectores)

Ventaja sobre correr los modulos por separado: usa una sola SparkSession y no
relee el parquet del disco entre el exporter y los embeddings (mas eficiente).

Los embeddings son OPCIONALES (son un paso pesado, ~varios minutos) y se generan
una sola vez. Por defecto el ETL NO los genera; agregar --with-embeddings para
incluirlos.

Uso (dentro del contenedor de Spark, parado en /opt/spark):
    export PYTHONPATH=/opt/spark/src:/opt/spark/src/etl
    /opt/spark/bin/spark-submit src/etl/run_etl.py                    # rapido (sin embeddings)
    /opt/spark/bin/spark-submit src/etl/run_etl.py --with-embeddings  # flujo completo
"""
import argparse

from pyspark.sql import SparkSession

from loader import load_data
from cleaner import mrmusculo
from exporter import export_to_parquet
from embeddings import add_embeddings

# Rutas de entrada/salida (relativas a la raiz del proyecto)
RAW_HOTELS = "data/raw/Datafiniti_Hotel_Reviews.csv"
RAW_STATES = "data/raw/states.csv"
CLEAN_PARQUET = "data/processed/hotel_reviews.parquet"
EMBEDDINGS_PARQUET = "data/processed/hotel_reviews_embeddings.parquet"


def main(with_embeddings=False):
    spark = SparkSession.builder.appName("HotelReviews-ETL").getOrCreate()

    # 1) Carga de los dos CSV
    print("Cargando CSVs...")
    hotels_df, states_df = load_data(RAW_HOTELS, RAW_STATES)

    # 2) Limpieza + inner join + filtro de idioma + columnas derivadas
    print("Limpiando, cruzando y filtrando...")
    clean_df = mrmusculo(hotels_df, states_df)

    # 3) Guardar el parquet limpio
    print("Exportando parquet limpio...")
    export_to_parquet(clean_df, CLEAN_PARQUET)
    print(f"  -> {CLEAN_PARQUET}")

    # 4) Embeddings: paso PESADO y opcional (los vectores se generan una sola vez).
    if with_embeddings:
        print("Generando embeddings (esto tarda varios minutos)...")
        emb_df = add_embeddings(clean_df)
        emb_df.write.mode("overwrite").parquet(EMBEDDINGS_PARQUET)
        print(f"  -> {EMBEDDINGS_PARQUET}")
    else:
        print("Embeddings OMITIDOS (usá --with-embeddings para generarlos).")

    print("ETL completo.")
    spark.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ejecuta el ETL completo.")
    parser.add_argument(
        "--with-embeddings",
        action="store_true",
        help="Generar tambien los embeddings (paso pesado; por defecto se omite).",
    )
    args = parser.parse_args()
    main(with_embeddings=args.with_embeddings)
