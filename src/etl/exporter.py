# exporter.py
"""
este script se encarga de exportar los datos limpios a formato parquet para su uso en modelos o búsqueda.
Pasos:
1. Lee los datos limpios desde PostgreSQL
2. Exporta a formato parquet
"""
#Imports
import pandas as pd
from sqlalchemy import create_engine

def export_to_parquet(connection_string, output_path):
    """
    Lee los datos limpios desde PostgreSQL y los exporta a formato parquet.
    """
    # Paso 1: Lee los datos limpios desde PostgreSQL
    engine = create_engine(connection_string)
    df = pd.read_sql("SELECT * FROM hotel_reviews", engine)

    # Paso 2: Exporta a formato parquet
    df.to_parquet(output_path, index=False)
    print(f"Exportados {len(df)} registros a {output_path}")

if __name__ == "__main__":
    # Credenciales definidas en docker/docker-compose.yml
    connection_string = "postgresql://bigdata:bigdata123@localhost:5433/hotel_reviews"
    # Ruta de salida del parquet
    output_path = "data/processed/hotel_reviews.parquet"
    export_to_parquet(connection_string, output_path)