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
(estados), enriqueciendo cada reseña con su región y división. El dataset final
tiene **25.600 reseñas**.

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
> `torch` (CPU), `sentence-transformers`, `langdetect`, `pandas`, `pyarrow`,
> `pytest`, el cliente de Elasticsearch y el driver JDBC de PostgreSQL. Además
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
│   ├── ManualEjecucion.docx  # manual de ejecución (entregable)
│   ├── ManualEjecucion.pdf   # manual de ejecución (entregable, PDF)
│   ├── kibana_dashboard.ndjson  # dashboard de Kibana (5 visualizaciones)
│   └── diagrams/             # arquitectura, esquema Postgres, mappings ES (PNG)
├── src/
│   ├── etl/                  # carga y preprocesamiento (loader, cleaner, exporter, embeddings, run_etl)
│   ├── db/                   # PostgreSQL + Elasticsearch (es_setup, es_indexer, postgres_writer, main_materializacion, schemas/)
│   ├── search/               # generación de consultas para búsqueda semántica (generate_query)
│   └── models/               # modelos de predicción Spark ML (pipeline, logistic_regression, random_forest, main_models, resultados.ipynb)
└── tests/                    # pruebas unitarias (pytest)
```

## Requisitos previos

- **Docker Desktop** (se recomienda asignarle al menos **8–12 GB de RAM**).
- Colocar los CSV en `data/raw/`: `Datafiniti_Hotel_Reviews.csv` y `states.csv`
  (no van en el repo).
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

Interfaces web una vez levantado:

- Spark UI → http://localhost:8080
- Elasticsearch → http://localhost:9200
- Kibana → http://localhost:5601

---

## 1. Carga, preprocesamiento y embeddings (ETL)

> Módulos en `src/etl/`. Genera los datos limpios y los vectores.

Entrar al contenedor de Spark:

```bash
docker exec -it spark bash
cd /opt/spark
export PYTHONPATH=/opt/spark/src:/opt/spark/src/etl
```

Programa principal del ETL. Sin la bandera corre rápido (sin embeddings); los
embeddings son un paso pesado de una sola vez:

```bash
/opt/spark/bin/spark-submit src/etl/run_etl.py                    # carga + limpieza + parquet limpio
/opt/spark/bin/spark-submit src/etl/run_etl.py --with-embeddings  # además genera los embeddings
```

Flujo: `loader` (lee CSV) → `cleaner` (limpia + join + filtro de idioma + crea
`review_full_text` y `sentiment`) → `exporter` (parquet limpio) → `embeddings`
(agrega la columna `embedding` de 384 dims con un `pandas_udf` de Spark).

## 2. Materialización en PostgreSQL y Elasticsearch

> Módulos en `src/db/`. Escribe los datos en ambas bases.

Programa principal que crea los índices de ES, indexa los documentos y escribe las
tablas de Postgres (las fuentes crudas antes del cruce y el dataset final después):

```bash
export PYTHONPATH=/opt/spark
/opt/spark/bin/spark-submit src/db/main_materializacion.py
```

- **PostgreSQL:** tabla `hotel_reviews` (dataset cruzado) más `hotels_raw` y
  `states` (fuentes antes del cruce). Esquema en
  `src/db/schemas/postgres_schema.sql`. Validación con SQL en el puerto `5433`.
- **Elasticsearch:** dos índices, `hotel-reviews-bm25` (sin vectores) y
  `hotel-reviews-semantic` (campo `embedding` de 384 dims, similitud coseno). Los
  *mappings* se definen en `src/db/es_setup.py`.

## 3. Modelos de predicción (Spark ML)

> Módulos en `src/models/`. Variable objetivo: `sentiment`.

Se entrenan **dos modelos** con Spark MLlib (Regresión Logística y Random Forest)
usando como features el TF-IDF del texto y el rating, y se comparan sus resultados.

```bash
export PYTHONPATH=/opt/spark
/opt/spark/bin/spark-submit src/models/main_models.py
```

El análisis comparativo de resultados está en `src/models/resultados.ipynb`.

## 4. Búsqueda semántica y dashboards (Kibana)

- Búsqueda semántica: `python3 src/search/generate_query.py` genera dos consultas
  (BM25 sin vectores y kNN con vectores) para pegar en Kibana → Dev Tools y comparar
  cuál funciona mejor.
- **Dashboard** con 5 visualizaciones: importar `docs/kibana_dashboard.ndjson` en
  Kibana → Stack Management → Saved Objects → Import.

---

## Pruebas unitarias

Las pruebas (`tests/`) verifican que la carga y el preprocesamiento funcionan
correctamente. Se ejecutan **dentro del contenedor de Spark**.

```bash
docker exec -it spark bash
cd /opt/spark

# pyspark vive en $SPARK_HOME/python (no es un paquete pip), hay que agregarlo al path:
export PYTHONPATH=/opt/spark/src:/opt/spark/python:$(ls /opt/spark/python/lib/py4j-*-src.zip)

python3 -m pytest tests/ -v
```

Resultado esperado: **15 pruebas en verde** (loader, cleaner, exporter y
embeddings). Las pruebas de embeddings usan el modelo real; si `sentence-transformers`
no estuviera instalado, se saltan automáticamente (`importorskip`).

## Notas técnicas

- **Modelo de embeddings:** `all-MiniLM-L6-v2` (Hugging Face), 384 dimensiones,
  entrenado en inglés (las reseñas están en inglés). Pequeño y apto para correr
  localmente. Los vectores se normalizan (`normalize_embeddings=True`) para
  facilitar la búsqueda por similitud coseno.
- **Filtro de idioma:** con `langdetect` se descartan las reseñas que no están en
  inglés (~1%) y los textos sin sentido, porque el modelo de embeddings solo
  funciona bien con inglés.
- **Chunking:** se consideró, pero **no se aplicó**: las reseñas son cortas y no
  superan el límite de tokens del modelo, así que no hay truncamiento que mitigar.
- **Datos no versionados:** los CSV de `data/raw/` y los parquet de
  `data/processed/` están en `.gitignore` (archivos pesados). Para compartirlos se
  usa un `.tar.gz` o un enlace a Drive/OneDrive.
- **Parquet en nanosegundos:** si un parquet fue escrito por pandas/pyarrow, Spark
  3.5 no lo lee por defecto. Por eso el parquet limpio se **genera con Spark** antes
  de generar embeddings.

## Equipo

- **Andrés** — ETL: carga, preprocesamiento, embeddings y pruebas unitarias.
- **Liz** — Modelos de predicción (Spark ML) y análisis de resultados.
- **Fer** — Materialización (PostgreSQL + Elasticsearch), búsqueda semántica y
  dashboards de Kibana.
