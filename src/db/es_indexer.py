from pyspark.sql import SparkSession
from elasticsearch import Elasticsearch, helpers
import math
import pandas as pd

def load_to_elasticsearch():
    print("Iniciando sesión de Spark...")
    spark = SparkSession.builder.appName("ElasticSearch-Indexer").getOrCreate()

    parquet_path = "data/processed/hotel_reviews_embeddings.parquet"
    print(f"Leyendo datos de {parquet_path}...")
    df = spark.read.parquet(parquet_path)

    print("Convirtiendo datos para inyección...")
    
    pandas_df = df.toPandas()
    pandas_df = pandas_df.astype(object).where(pd.notna(pandas_df), None)
    records = pandas_df.to_dict(orient="records")

    es = Elasticsearch("http://elasticsearch:9200")

    def bm25_generator(data):
        for row in data:
            doc = row.copy()
            if "embedding" in doc:
                del doc["embedding"]
            
            yield {
                "_index": "hotel-reviews-bm25",
                "_source": doc
            }

    def semantic_generator(data):
        for row in data:
            doc = row.copy()
            if "embedding" in doc and hasattr(doc["embedding"], "tolist"):
                doc["embedding"] = doc["embedding"].tolist()
                
            yield {
                "_index": "hotel-reviews-semantic",
                "_source": doc
            }


    print("Inyectando datos en índice BM25...")
    helpers.bulk(es, bm25_generator(records))
    print("Índice BM25 poblado con éxito")

    print("Inyectando datos en índice Semántico...")
    helpers.bulk(es, semantic_generator(records))
    print("Índice Semántico poblado con éxito")

    spark.stop()

if __name__ == "__main__":
    load_to_elasticsearch()