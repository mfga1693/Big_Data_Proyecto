# exporter.py
"""
este script se encarga de guardar el DataFrame limpio en formato parquet.
Pasos:
1. Recibe el DataFrame ya limpio y transformado
2. Lo guarda en formato parquet en la carpeta data/processed
"""
# Imports, no necesita ninguno extra, Spark ya viene del DataFrame

def export_to_parquet(df, output_path):
    """
    Guarda el DataFrame en formato parquet.
    """
    # Paso 1: Guarda el DataFrame en parquet
    df.write.mode("overwrite").parquet(output_path)
    print(f"Exportados {df.count()} registros a {output_path}")

if __name__ == "__main__":
    from loader import load_data
    from cleaner import mrmusculo
    # Cargar y limpiar datos
    hotels_df, states_df = load_data("data/raw/Datafiniti_Hotel_Reviews.csv", "data/raw/states.csv")
    df = mrmusculo(hotels_df, states_df)
    # Exportar a parquet
    export_to_parquet(df, "data/processed/hotel_reviews.parquet")