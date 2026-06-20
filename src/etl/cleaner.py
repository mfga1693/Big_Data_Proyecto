# cleaner.py
"""
este script se encarga de limpiar los datos antes de usarlo para modelos o búsqueda.
Pasos:
1. Eliminar filas donde el texto de la reseña esté vacío (no puedes analizar lo que no existe)
2. Elimina duplicados
3. limpia espacios en blanco al inicio y al final de los textos
4. Convertir el rating a número (en el CSV viene como texto a veces)
"""
#Imports
import pandas as pd

def mrmusculo(df):
    """
    Limpia el DataFrame de reseñas.
    """
    #Paso 1: eliminar filas donde el texto de la reseña esté vacío
    df = df[df['review_text'].notna()]
    #Paso 2: Elimina duplicados
    df = df.drop_duplicates()
    #Paso 3: limpia espacios en blanco al inicio y al final de los textos
    df['review_text'] = df['review_text'].str.strip()
    #Paso 4: Convertir el rating a número
    df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
    return df

if __name__ == "__main__":
    import pandas as pd
    # Prueba rápida con el CSV
    df = pd.read_csv("data/raw/Datafiniti_Hotel_Reviews.csv")
    df_limpio = mrmusculo(df)
    print(f"Registros después de limpiar: {len(df_limpio)}")