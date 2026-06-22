# conftest.py
# Configuracion comun para las pruebas unitarias con pytest.
#
# - Agrega `src/` al path para poder importar los modulos del ETL.
# - Provee una SparkSession local reutilizable para toda la sesion de pruebas.
import os
import sys

import pytest

# Permite importar `from etl import ...` en las pruebas.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))


@pytest.fixture(scope="session")
def spark():
    """SparkSession local, compartida por todas las pruebas (se crea una vez)."""
    from pyspark.sql import SparkSession

    session = (
        SparkSession.builder.master("local[2]")
        .appName("etl-tests")
        # bindAddress/host a loopback: evita fallos de resolucion de hostname.
        .config("spark.driver.host", "127.0.0.1")
        .config("spark.driver.bindAddress", "127.0.0.1")
        .config("spark.sql.shuffle.partitions", "2")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )
    session.sparkContext.setLogLevel("ERROR")
    yield session
    session.stop()
