# Guía de ejecución — paso a paso (for dummies)

Cómo correr el proyecto de cero. Seguí los pasos en orden.

> Estado actual: el **ETL + embeddings + pruebas + modelos** ya se pueden correr.
> La **materialización (Postgres/Elasticsearch) y los dashboards de Kibana** son
> parte de Fer (el código de Postgres y ES ya está; faltan los dashboards).

---

## 0. Requisitos (una sola vez)

1. Tener **Docker Desktop** instalado y **abierto** (esperá a que diga "running").
   Recomendado: asignarle 8–12 GB de RAM (Settings → Resources).
2. Tener el repo clonado y estar parado en su carpeta:
   ```bash
   cd /ruta/a/Big_Data_Proyecto
   ```
3. **Poner los CSV de datos** en `data/raw/` (NO están en GitHub por ser pesados):
   - `data/raw/Datafiniti_Hotel_Reviews.csv`
   - `data/raw/states.csv`

---

## 1. Levantar el entorno (Docker)

Desde la carpeta `docker/`:

```bash
cd docker
docker compose build spark      # construye la imagen (1ra vez tarda: baja torch + el modelo)
docker compose up -d            # levanta Spark, Postgres, Elasticsearch y Kibana
docker ps                       # deberías ver los contenedores "Up"
```

Interfaces (cuando estén arriba):
- Spark UI → http://localhost:8080
- Elasticsearch → http://localhost:9200
- Kibana → http://localhost:5601

---

## 2. Entrar al contenedor de Spark

```bash
docker exec -it spark bash
```

El prompt cambia a `spark@xxxx:/opt/spark/work-dir$` → ya estás **dentro**.
Ubicate y prepará el path:

```bash
cd /opt/spark
export PYTHONPATH=/opt/spark/src:/opt/spark/src/etl
```

---

## 3. Correr el ETL

La forma recomendada es el programa principal **`run_etl.py`** (todo en un comando):

```bash
# Rápido: cargar + limpiar + cruzar + exportar el parquet limpio (SIN embeddings)
/opt/spark/bin/spark-submit src/etl/run_etl.py
```

✅ Termina con `Exportados 25600 registros ...` y `ETL completo.`

Los **embeddings son un paso pesado y de una sola vez**, por eso NO se generan por
defecto. La **primera vez** (o si cambian los datos), corré con la bandera:

```bash
# Flujo completo: lo anterior + generar embeddings (tarda varios minutos)
/opt/spark/bin/spark-submit src/etl/run_etl.py --with-embeddings
```

✅ Genera también `data/processed/hotel_reviews_embeddings.parquet` (columna `embedding`, 384 dims).

### Alternativa: módulos por separado

`run_etl.py` es solo un atajo; cada módulo corre suelto:

```bash
/opt/spark/bin/spark-submit src/etl/exporter.py     # loader + cleaner + exporter (parquet limpio)
/opt/spark/bin/spark-submit src/etl/embeddings.py   # genera los embeddings
```

> `exporter.py` ya llama internamente a `loader` y `cleaner`; no hay que correrlos aparte.

---

## 4. Correr las pruebas unitarias

pyspark vive en `$SPARK_HOME/python`, hay que sumarlo al path:

```bash
export PYTHONPATH=/opt/spark/python:$(ls /opt/spark/python/lib/py4j-*-src.zip):/opt/spark/src
python3 -m pytest tests/ -v
```

✅ Esperado: **15 passed**

---

## 5. Modelos de predicción (parte de Liz)

> Nota: `main_models.py` está en la raíz del repo, que NO se monta en el
> contenedor. Para correrlo dentro de Docker hay que moverlo a `scripts/` (carpeta
> montada). Una vez movido:

```bash
cd /opt/spark
export PYTHONPATH=/opt/spark
/opt/spark/bin/spark-submit scripts/main_models.py
```

Entrena Logistic Regression y Random Forest e imprime métricas (Accuracy, F1, AUC-ROC).

---

## 6. Materialización + búsqueda semántica + dashboards (parte de Fer)

**PostgreSQL** (el driver JDBC ya viene en la imagen):

```bash
/opt/spark/bin/spark-submit src/db/postgres_writer.py
```

**Elasticsearch** (crear índices + cargar datos):

```bash
python3 src/db/es_setup.py     # crea los 2 índices (BM25 + semántico) con sus mappings
python3 src/db/es_indexer.py   # carga los 25.600 documentos a ambos índices
```

**Búsqueda semántica** (genera la query kNN para pegar en Kibana Dev Tools):

```bash
python3 src/search/generate_query.py
```

⏳ **Dashboards de Kibana** (≥5 visualizaciones): se crean en la UI de Kibana
(http://localhost:5601). Pendiente.

---

## 7. Apagar el entorno

```bash
exit                 # salir del contenedor
docker compose down  # apagar los contenedores (desde la carpeta docker/)
```

---

## Problemas comunes

- **`spark-submit: command not found`** → usá la ruta completa: `/opt/spark/bin/spark-submit`.
- **`No module named 'pyspark'`** (al correr pytest) → te faltó el `export PYTHONPATH` del paso 4.
- **`No module named 'cleaner'`** (al correr el ETL) → te faltó `:/opt/spark/src/etl` en el PYTHONPATH (paso 2).
- **`No module named 'langdetect'` / `'elasticsearch'`** → la imagen no se reconstruyó; corré `docker compose build spark`.
- **`docker: command not found`** dentro del contenedor → ya estás dentro; ese comando es de tu Mac.
- **`schema.sql` de Postgres no se aplicó** → el init solo corre con volumen vacío; recreá: `docker compose down -v` y `docker compose up -d`.
