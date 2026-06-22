from pyspark.sql import SparkSession

def load_to_postgres():
    spark = SparkSession.builder.appName("Postgres-Writer").getOrCreate()
    df = spark.read.parquet("data/processed/hotel_reviews_embeddings.parquet")
    df = df.drop("embedding")  # el vector no va en la base relacional

    (df.write
        .format("jdbc")
        .option("url", "jdbc:postgresql://postgres:5432/hotel_reviews")
        .option("dbtable", "hotel_reviews")
        .option("user", "bigdata")
        .option("password", "bigdata123")
        .option("driver", "org.postgresql.Driver")
        .option("truncate", "true")
        .mode("overwrite")
        .save())

    print(f"Escritos {df.count()} registros en PostgreSQL (tabla hotel_reviews)")
    spark.stop()

if __name__ == "__main__":
    load_to_postgres()