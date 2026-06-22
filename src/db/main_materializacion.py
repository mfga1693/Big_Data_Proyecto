"""
Programa principal de materializacion.

Orquesta la escritura de los datos procesados (el parquet del ETL) en las dos
bases de datos del proyecto:
  1. Elasticsearch: crea los indices (BM25 sin vectores + semantico con vectores).
  2. Elasticsearch: carga (indexa) los documentos en ambos indices.
  3. PostgreSQL: escribe los datos en la tabla relacional.

Usa los modulos de materializacion ya existentes (no reimplementa nada).

Uso (dentro del contenedor de Spark, parado en /opt/spark):
    export PYTHONPATH=/opt/spark
    /opt/spark/bin/spark-submit src/db/main_materializacion.py
"""
from src.db.es_setup import setup_elasticsearch
from src.db.es_indexer import load_to_elasticsearch
from src.db.postgres_writer import load_to_postgres


def main():
    print("=== 1/3  Creando indices en Elasticsearch (BM25 + semantico) ===")
    setup_elasticsearch()

    print("=== 2/3  Indexando documentos en Elasticsearch ===")
    load_to_elasticsearch()

    print("=== 3/3  Escribiendo datos en PostgreSQL ===")
    load_to_postgres()

    print("\nMaterializacion completa: PostgreSQL + Elasticsearch.")


if __name__ == "__main__":
    main()
