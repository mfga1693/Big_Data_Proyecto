# loader.py
"""
este script se encarga de leer el archivo csv y cargarlo a la base de datos postgreSQL.
Pasos:
1. Lee el CSV con pandas
2. Selecciona y renombra las columnas necesarias
3. Se conecta a PostgreSQL
4. Inserta los datos en la tabla hotel_reviews
"""
#Imports
import pandas as pd
from sqlalchemy import create_engine

def load_csv_to_postgres(csv_path, connection_string):
    """
    Lee el CSV y lo carga a PostgreSQL.
    """
    # Paso 1: Lee el CSV con pandas
    df = pd.read_csv(csv_path)
    # Paso 2: Selecciona y renombra las columnas necesarias
    df = df[['name', 'city', 'province', 'reviews.date', 'reviews.rating', 'reviews.text', 'reviews.title', 'reviews.username']]
    df.columns = ['hotel_name', 'city', 'province', 'review_date', 'rating', 'review_text', 'review_title', 'username']
    # Paso 3: Se conecta a PostgreSQL
    engine = create_engine(connection_string)
    # Paso 4: Inserta los datos en la tabla hotel_reviews
    df.to_sql('hotel_reviews', engine, if_exists='append', index=False) 

    print(f"Cargados {len(df)} registros en hotel_reviews")

if __name__ == "__main__":
    #Ruta al CSV original 
    csv_path = "data/raw/Datafiniti_Hotel_Reviews.csv"
    # Credenciales definidas en docker/docker-compose.yml
    connection_string = "postgresql://bigdata:bigdata123@localhost:5433/hotel_reviews"
    load_csv_to_postgres(csv_path, connection_string)


