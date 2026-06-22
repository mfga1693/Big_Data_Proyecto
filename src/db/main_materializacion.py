"""Materialización de datos para el proyecto de búsqueda semántica.
Este script se encarga de:
1. Crear los índices necesarios en Elasticsearch (BM25 y semántico)
2. Indexar los documentos en ambos índices de Elasticsearch
3. Escribir los datos en PostgreSQL para consultas analíticas
Uso (dentro del contenedor de Spark, en /opt/spark):
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
