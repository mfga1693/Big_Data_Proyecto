# embeddings.py
#
# Objetivo: generar un vector (embedding) por cada resena, dentro del flujo de Spark.


# ===== BLOQUE 1: imports y configuracion (HF_HOME) =====
import os
import pandas as pd
from pyspark.sql import SparkSession

# Funciones para manipular columnas del DataFrame:
from pyspark.sql.functions import coalesce, col, lit, pandas_udf
# coalesce(a, b): si 'a' es nulo usa 'b'. Lo usamos para reemplazar
#                 review_full_text nulo por una cadena vacia "".
# col("x"):       apunta a la columna llamada "x".
# lit(""):        convierte un valor fijo ("") en algo que Spark trata como columna.
# pandas_udf:     crea una funcion propia (UDF) que Spark corre en paralelo por
#                 lotes; aqui la usamos para aplicar el modelo de Hugging Face.

# Tipos para declarar la columna de salida (el vector):
from pyspark.sql.types import ArrayType, FloatType
# La columna 'embedding' sera una lista de decimales ->
# ArrayType(FloatType()) = lista (Array) de decimales (Float).

# El "estante": empieza vacío. Aquí se guardará el modelo una vez cargado.
#De alguan forma, el modelo es como una maquina que construimos a partir de un plano (la arquitectura del modelo) y unos materiales (los pesos preentrenados).
#Cargar el modelo es como construir la maquina, y luego la usamos para generar los embeddings. El estante es donde guardamos la maquina para no tener que reconstruirla cada vez que queremos usarla.

# Carpeta de caché para descargar el modelo de Hugging Face.
# Debe ser escribible (este era tu bloqueo en el contenedor)
os.environ.setdefault("HF_HOME", "/tmp/hf_cache") #esta dixce donde va el modelo, /tmp es escribible en el contenedor
os.makedirs(os.environ["HF_HOME"], exist_ok=True) #crea la carpeta si no existe


# ===== BLOQUE 2: cargar el modelo (singleton) =====
_MODEL = None


def get_model(model_name="all-MiniLM-L6-v2"):
    """Carga el modelo UNA sola vez por proceso y lo reutiliza (patrón singleton)."""
    global _MODEL #indica que vamos a usar la variable global _MODEL, no una local
    if _MODEL is None:                            # ¿el estante está vacío?
        from sentence_transformers import SentenceTransformer #import perezoso, lo hacemos dentro de la función para no cargarlo si no es necesario
        _MODEL = SentenceTransformer(model_name)  # construye y guarda en el estante
    return _MODEL                                 # entrega lo que hay en el estante


# ===== BLOQUE 3: el pandas_udf que vectoriza (el motor) =====
@pandas_udf(ArrayType(FloatType())) #decorador que indica que esta funcion es una UDF de Spark que devuelve una columna de tipo ArrayType(FloatType())
def embed_udf(textos: pd.Series) -> pd.Series:
    """
    Recibe un LOTE de textos y devuelve un LOTE de vectores (uno por texto). Esto es en parte lo
    que hace que pandas_udf sea tan eficiente: en lugar de procesar un texto a la vez, procesa
    un lote completo, lo que es mucho más rápido.
    """
    modelo = get_model()                          # agarra el modelo del estante, la primera vez lo crea despues reutiliza
    vectores = modelo.encode(
        textos.tolist(),                          # la Serie de pandas -> lista de Python
        batch_size=64,
        show_progress_bar=False,
        normalize_embeddings=True,  # cada vector se ajusta a largo 1: así la similitud
                                    # compara solo la DIRECCIÓN (el significado), no el
                                    # largo del texto. Ideal para búsqueda por coseno.
    )
    # Convertimos cada vector a lista de floats normales y lo devolvemos como Serie:
    #entra un lote de textos → el modelo los convierte en vectores → devolvemos un lote de listas de 384 floats
    return pd.Series([[float(x) for x in v] for v in vectores])

# ===== BLOQUE 4: pegar la columna 'embedding' al DataFrame =====
def add_embeddings(df, text_col="review_full_text", output_col="embedding"):
    """Agrega la columna de embeddings a un DataFrame de Spark. Aun aqui no se ejecuta nada, solo preparamos la receta."""
    # Si el texto es nulo, lo reemplazamos por "" para que el modelo no falle (los 285 nulos):
    texto_seguro = coalesce(col(text_col), lit(""))
    # withColumn crea la columna nueva aplicando el UDF. Spark es perezoso:
    # esto solo anota la receta; el cálculo real ocurre al guardar/contar.
    return df.withColumn(output_col, embed_udf(texto_seguro))

# ===== BLOQUE 5: función principal (lee parquet limpio -> escribe parquet con vectores) =====
def generate_embeddings(input_path, output_path, spark=None):
    """Lee el parquet limpio, genera embeddings y guarda el parquet con vectores."""
    creamos_spark = False
    if spark is None:                                   # si no nos pasan Spark, lo creamos
        spark = SparkSession.builder.appName("HotelReviews-Embeddings").getOrCreate()
        creamos_spark = True

    df = spark.read.parquet(input_path)                 # lee el parquet limpio
    df_emb = add_embeddings(df)                          # agrega la columna 'embedding' (Bloque 4)
    df_emb.write.mode("overwrite").parquet(output_path)  # AQUÍ sí se ejecuta todo (Spark perezoso)

    print(f"Embeddings generados para {df_emb.count()} reseñas -> {output_path}")

    if creamos_spark:                                   # solo cerramos lo que nosotros abrimos
        spark.stop()
    return output_path


if __name__ == "__main__":
    generate_embeddings(
        "data/processed/hotel_reviews.parquet",
        "data/processed/hotel_reviews_embeddings.parquet",
    )