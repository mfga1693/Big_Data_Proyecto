# Dockerfile — Spark + librerias de ML para generar embeddings
#
# Parte de la imagen oficial de Spark (trae Java + Spark + Python) y le agrega
# las librerias que el codigo de embeddings necesita: torch (CPU),
# sentence-transformers, pandas y pyarrow. Ademas pre-descarga el modelo
# all-MiniLM-L6-v2 DENTRO de la imagen, para que en tiempo de ejecucion no haya
# que bajarlo (esto evita el problema de permisos de HuggingFace que teniamos).

FROM apache/spark:3.5.4

# Necesitamos privilegios de root para instalar paquetes de sistema/Python.
USER root

# Carpeta de cache de HuggingFace (escribible). El modelo se baja aqui en el build.
# El codigo usa os.environ.setdefault("HF_HOME", ...), asi que respeta este valor.
ENV HF_HOME=/opt/hf_cache

# 1) torch en version CPU (mas liviano; no necesitamos GPU para inferencia chica)
# 2) el resto de dependencias desde requirements.txt
COPY docker/requirements.txt /tmp/requirements.txt
RUN python3 -m pip install --no-cache-dir --upgrade pip && \
    python3 -m pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    python3 -m pip install --no-cache-dir -r /tmp/requirements.txt

# Pre-descargar el modelo dentro de la imagen y dejar la cache accesible a todos.
RUN python3 -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')" && \
    chmod -R 777 /opt/hf_cache

# Driver JDBC de PostgreSQL para que Spark pueda escribir a la base relacional.
# Se hornea en la imagen (en los jars de Spark) para no depender de --packages/Ivy.
RUN python3 -c "import urllib.request; urllib.request.urlretrieve('https://repo1.maven.org/maven2/org/postgresql/postgresql/42.7.3/postgresql-42.7.3.jar', '/opt/spark/jars/postgresql-42.7.3.jar')"

# Volvemos al usuario no-root que usa la imagen de Spark.
USER spark
