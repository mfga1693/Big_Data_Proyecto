from pyspark.sql import SparkSession

PG = {
    "url": "jdbc:postgresql://postgres:5432/hotel_reviews",
    "user": "bigdata",
    "password": "bigdata123",
    "driver": "org.postgresql.Driver",
}


def _write(df, table):
    (df.write
        .format("jdbc")
        .option("url", PG["url"])
        .option("dbtable", table)
        .option("user", PG["user"])
        .option("password", PG["password"])
        .option("driver", PG["driver"])
        .mode("overwrite")
        .save())
    print(f"  -> tabla '{table}': {df.count()} filas")


def load_to_postgres():
    spark = SparkSession.builder.appName("Postgres-Writer").getOrCreate()

    print("Escribiendo fuentes (antes del cruce)...")

    hotels_raw = spark.read.csv("data/raw/Datafiniti_Hotel_Reviews.csv",
                                header=True, inferSchema=True)
    hotels_raw = hotels_raw.toDF(*[c.replace(".", "_") for c in hotels_raw.columns])
    _write(hotels_raw, "hotels_raw")

    states = spark.read.csv("data/raw/states.csv", header=True, inferSchema=True)
    states = states.toDF(*[c.strip().replace(" ", "_").lower() for c in states.columns])
    _write(states, "states")

    print("Escribiendo dataset cruzado (después del cruce)...")
    df = spark.read.parquet("data/processed/hotel_reviews_embeddings.parquet").drop("embedding")
    _write(df, "hotel_reviews")

    print("Materialización en PostgreSQL completa.")
    spark.stop()


if __name__ == "__main__":
    load_to_postgres()