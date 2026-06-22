# Diagramas

Carpeta para los diagramas del proyecto (los pide el rubro: "diagramas, mappings y
esquemas para ambos motores de bases de datos").

Diagramas esperados:

- `arquitectura.png` — diagrama general del sistema
  (CSVs → Spark ETL → parquet → PostgreSQL + Elasticsearch → Kibana / modelos).
- `postgres_schema.png` — diagrama entidad-relación de PostgreSQL
  (basado en `src/db/schemas/postgres_schema.sql`).
- `es_mappings.png` — diagrama de los mappings de Elasticsearch
  (índice BM25 sin vectores + índice semántico con `dense_vector` 384D;
  basado en `src/db/es_setup.py`).
