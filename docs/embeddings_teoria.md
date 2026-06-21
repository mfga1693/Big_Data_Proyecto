# Embeddings — Notas de teoría

> Notas de apoyo para entender y explicar la parte de generación de vectores
> (embeddings) del proyecto. Autor: Andrés.

## 1. Qué es un embedding (y por qué lo necesitamos)

Una computadora no "entiende" texto, entiende números. Un **embedding** es
convertir un texto en una lista de números (un *vector*) que captura su
*significado*. La clave es que **textos con significado parecido quedan cerca en
ese espacio numérico**, aunque usen palabras distintas.

Ejemplo: "the room was filthy" y "very dirty bedroom" no comparten casi
palabras, pero un buen embedding las pone *cerca* porque significan algo
parecido.

Esto es distinto de la búsqueda tradicional (TF-IDF, que cuenta palabras). Si
buscás "dirty" con TF-IDF, no encuentra "filthy". Con embeddings sí, porque
compara *significado*. Eso es la **búsqueda semántica** que pide el proyecto, y
por eso los vectores van al índice "con vectores" de Elasticsearch.

## 2. El modelo que genera los vectores

No entrenamos nada nosotros. Usamos un modelo ya entrenado de Hugging Face:
**`all-MiniLM-L6-v2`**, a través de la librería `sentence-transformers`.

Datos clave que hay que poder explicar:

- Cada texto que le entra → sale un vector de **384 números** (384 dimensiones).
  Siempre 384, sin importar el largo del texto.
- Es un modelo **en inglés** (las reseñas son en inglés → correcto) y
  **pequeño**, ideal para correr local.
- Se descarga la primera vez desde internet a una carpeta de caché (`HF_HOME`).
  Si el contenedor no tiene permiso de escritura ahí, falla. Se arregla
  apuntando `HF_HOME` a una carpeta escribible.

## 3. Dónde encaja en el pipeline

```
loader (lee CSVs)
  → cleaner (limpia + join + crea review_full_text)
    → exporter (guarda parquet limpio)
      → embeddings (lee ese parquet, agrega columna `embedding`, guarda otro parquet)
```

Es decir: entra el parquet limpio, y el trabajo es **agregar una columna nueva
llamada `embedding`** donde cada fila tenga el vector de su `review_full_text`.
Ese parquet con vectores es lo que se carga en Elasticsearch.

## 4. Por qué un UDF de Spark

Spark reparte los datos en **particiones** y las procesa en paralelo. Un **UDF**
(*user-defined function*) es una función propia que Spark ejecuta en paralelo
sobre cada partición. Es la forma de meter código propio (como llamar al modelo)
dentro del flujo de Spark.

Usamos un **`pandas_udf`**, una versión más eficiente: en vez de pasar el texto
fila por fila, Spark le pasa **lotes** de textos (como una columna de pandas), el
modelo los vectoriza de golpe (más rápido), y devuelve la columna de vectores.

Detalle importante: **el modelo se carga UNA vez por worker, no por fila**.
Cargar el modelo es caro (segundos); si se cargara por cada reseña tardaría
muchísimo. Por eso se usa un patrón "singleton": la primera vez se carga y se
guarda en memoria, las siguientes lo reutiliza.

### Cuidados prácticos

- **Textos nulos**: hay 285 reseñas con `review_full_text` nulo. El modelo falla
  con nulos, así que antes de vectorizar se reemplazan por texto vacío `""`.
- **Chunking**: no aplica aquí (reseñas cortas), pero se menciona como
  "considerado". Solo haría falta si los textos superaran el límite del modelo
  (~256 word-pieces en este modelo) y se truncaran, perdiendo contexto.

## 5. Estructura del archivo `embeddings.py`

En orden:

1. Imports
2. Configurar `HF_HOME` (carpeta escribible para la caché del modelo)
3. Función que carga el modelo (singleton)
4. El `pandas_udf` que vectoriza
5. Función que agrega la columna `embedding` al DataFrame
6. Función principal que lee el parquet limpio y escribe el parquet con vectores
