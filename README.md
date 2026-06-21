# Proyecto Big Data — Análisis de Reseñas de Hoteles

Proyecto de ETL, modelos de predicción y búsqueda semántica sobre reseñas de
hoteles. Los datos se cargan y transforman con **Spark**, se generan **embeddings**
con Hugging Face, se materializan en **PostgreSQL** y **Elasticsearch**, y se
exploran con **Kibana**.

## Objetivos

- **Objetivo predictivo:** predecir el `sentiment` de una reseña (1 = positivo si
  la calificación es ≥ 4, 0 = negativo) a partir del texto y otras variables.
- **Objetivo de búsqueda semántica:** buscar reseñas por significado (no solo por
  palabras) usando vectores generados del campo `review_full_text` (texto en
  inglés), almacenados en un índice vectorial de Elasticsearch.

## Fuentes de datos

| Fuente | Descripción | Llave de cruce |
|--------|-------------|----------------|
| `Datafiniti_Hotel_Reviews.csv` | Reseñas de hoteles en EE.UU. (texto, rating, fecha, ubicación). | `province` |
| `states.csv` | Catálogo de estados de EE.UU. (nombre, código, región, división). | `State Code` |

El cruce se hace con un **inner join** entre `province` (reseñas) y `State Code`
(estados), enriqueciendo cada reseña con su región y división.

## Stack y arquitectura

Todo corre en contenedores definidos en `docker/docker-compose.yml`:

| Servicio | Imagen | Puerto(s) | Rol |
|----------|--------|-----------|-----|
| `spark` | propia (`docker/Dockerfile`) | 8080, 7077 | Spark master + librerías de ML (torch, sentence-transformers) |
| `spark-worker` | misma imagen propia | — | Worker de Spark |
| `postgres` | postgres:15 | 5433 → 5432 | Base relacional (`bigdata` / `bigdata123` / `hotel_reviews`) |
| `elasticsearch` | elasticsearch:8.13.0 | 9200, 9300 | Índices con y sin vectores |
| `kibana` | kibana:8.13.0 | 5601 | Búsqueda semántica y dashboards |

> La imagen de Spark se construye a partir de `apache/spark:3.5.4` y agrega
> `torch` (CPU), `sentence-transformers`, `pandas`, `pyarrow` y `pytest`. Además
> **pre-descarga el modelo `all-MiniLM-L6-v2`** dentro de la imagen, para no
> depender de la red ni de permisos de escritura en tiempo de ejecución.

## Estructura del repositorio

```
Big_Data_Proyecto/
├── data/
│   ├── raw/                  # CSVs de entrada (no versionados)
│   └── processed/            # parquet generados (no versionados)
├── docker/
│   ├── Dockerfile            # imagen de Spark + librerías de ML
│   ├── docker-compose.yml    # definición de los 5 servicios
│   └── requirements.txt      # dependencias de Python de la imagen
├── docs/
│   └── embeddings_teoria.md  # notas teóricas de embeddings
├── scripts/
│   └── smoke_embeddings.py   # prueba rápida del modelo (20 reseñas)
├── src/
│   ├── etl/                  # carga y preprocesamiento (loader, cleaner, exporter, embeddings)
│   ├── features/             # feature engineering (TF-IDF) para ML
│   ├── models/               # modelos de predicción (Spark ML)
│   ├── db/                   # escritura a PostgreSQL
│   └── search/               # carga a Elasticsearch y búsqueda semántica
└── tests/                    # pruebas unitarias (pytest)
```

## Requisitos previos

- **Docker Desktop** (se recomienda asignarle al menos **8–12 GB de RAM**).
- No hace falta instalar Python, Java ni Spark en la máquina: todo vive en los
  contenedores.

## Puesta en marcha

Desde la carpeta `docker/`:

```bash
cd docker

# Construir la imagen de Spark (la primera vez tarda: baja torch + el modelo)
docker compose build spark

# Levantar todos los servicios
docker compose up -d

# Verificar que están arriba
docker ps
```

Para trabajar solo con el ETL alcanza con `spark` y `spark-worker`:

```bash
docker compose up -d spark spark-worker
```

Interfaces web una vez levantado:

- Spark UI → http://localhost:8080
- Elasticsearch → http://localhost:9200
- Kibana → http://localhost:5601

---

## 1. Carga, preprocesamiento y embeddings (ETL)

> Módulos en `src/etl/`. Esta sección genera los datos limpios y los vectores.

Entrar al contenedor de Spark y ubicarse en el directorio de trabajo:

```bash
docker exec -it spark bash
cd /opt/spark
```

**a) Generar el parquet limpio** (lee los CSV, limpia, cruza con estados y exporta):

```bash
/opt/spark/bin/spark-submit src/etl/exporter.py
# -> data/processed/hotel_reviews.parquet  (27.116 registros)
```

**b) (Opcional) Prueba de humo** del modelo de embeddings sobre 20 reseñas:

```bash
export PYTHONPATH=/opt/spark/src
/opt/spark/bin/spark-submit scripts/smoke_embeddings.py
# -> "Dimensiones del vector: 384"
```

**c) Generar los embeddings** de todas las reseñas (columna `embedding` de 384 dims):

```bash
export PYTHONPATH=/opt/spark/src
/opt/spark/bin/spark-submit src/etl/embeddings.py
# -> data/processed/hotel_reviews_embeddings.parquet
```

Flujo: `loader` (lee CSV) → `cleaner` (limpia + join + crea `review_full_text` y
`sentiment`) → `exporter` (parquet limpio) → `embeddings` (agrega la columna
`embedding` con un `pandas_udf` de Spark).

## 2. Materialización en PostgreSQL y Elasticsearch

> Módulos en `src/db/` y `src/search/`. Escribe los datos en las bases.

- **PostgreSQL:** se cargan las reseñas limpias (esquema en
  `src/db/schemas/postgres_schema.sql`). Validación con consultas SQL en el puerto
  `5433`.
- **Elasticsearch:** se crean **dos índices**:
  - uno **sin vectores** (búsqueda tradicional por texto),
  - uno **con vectores** (campo `embedding` de 384 dims para búsqueda semántica).
  - Los *mappings* de ambos índices se documentan en `src/search/`.

```bash
# Ejemplo (programa principal de materialización):
/opt/spark/bin/spark-submit src/db/load_postgres.py
/opt/spark/bin/spark-submit src/search/load_elasticsearch.py
```

## 3. Modelos de predicción (Spark ML)

> Módulos en `src/features/` y `src/models/`. Variable objetivo: `sentiment`.

Se entrenan **al menos dos modelos** con Spark MLlib (por ejemplo Regresión
Logística y Random Forest) usando como features el TF-IDF del texto y el rating,
y se comparan sus resultados.

```bash
/opt/spark/bin/spark-submit src/models/train.py
```

## 4. Búsqueda semántica y dashboards (Kibana)

- Consultas en Kibana **con vectores** (kNN sobre `embedding`) y **sin vectores**
  (match de texto), documentando cuál funciona mejor.
- **Dashboard** con al menos 5 visualizaciones (distribución de ratings, sentiment
  por estado/región, hoteles más reseñados, etc.).

---

## Pruebas unitarias

Las pruebas (`tests/`) verifican que la carga y el preprocesamiento funcionan
correctamente. Se ejecutan **dentro del contenedor de Spark**.

```bash
docker exec -it spark bash
cd /opt/spark

# pyspark vive en $SPARK_HOME/python (no es un paquete pip), hay que agregarlo al path:
export PYTHONPATH=/opt/spark/python:$(ls /opt/spark/python/lib/py4j-*-src.zip):/opt/spark/src

python3 -m pytest tests/ -v
```

Resultado esperado: **12 pruebas en verde** (cleaner, loader, exporter y
embeddings). Las pruebas de embeddings usan el modelo real; si `sentence-transformers`
no estuviera instalado, se saltan automáticamente (`importorskip`).

## Notas técnicas

- **Modelo de embeddings:** `all-MiniLM-L6-v2` (Hugging Face), 384 dimensiones,
  entrenado en inglés (las reseñas están en inglés). Pequeño y apto para correr
  localmente. Los vectores se normalizan (`normalize_embeddings=True`) para
  facilitar la búsqueda por similitud coseno.
- **Chunking:** se consideró, pero **no se aplicó**: las reseñas son cortas y no
  superan el límite de tokens del modelo, así que no hay truncamiento que mitigar.
- **HF_HOME:** la caché del modelo apunta a una carpeta escribible
  (`/opt/hf_cache` en la imagen). El código respeta la variable de entorno si ya
  está definida.
- **Datos no versionados:** los CSV de `data/raw/` y los parquet de
  `data/processed/` están en `.gitignore` (archivos pesados). Para compartirlos se
  usa un `.tar.gz` o un enlace a Drive/OneDrive.
- **Parquet en nanosegundos:** si un parquet fue escrito por pandas/pyarrow, Spark
  3.5 no lo lee por defecto. Por eso el parquet limpio se **regenera con Spark**
  (`exporter.py`) antes de generar embeddings.

## Equipo

- **Andrés** — ETL: carga, preprocesamiento, embeddings y pruebas unitarias.
- **Liz** — Modelos de predicción (Spark ML) y análisis de resultados.
- **Fer** — Materialización (PostgreSQL + Elasticsearch), búsqueda semántica y
  dashboards de Kibana.
