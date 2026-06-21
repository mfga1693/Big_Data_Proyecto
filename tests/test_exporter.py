# test_exporter.py
# Autor: Andres
#
# Pruebas unitarias del modulo exporter (funcion export_to_parquet): que escriba
# el DataFrame en Parquet y que al leerlo de vuelta los datos sean los mismos
# (prueba de "ida y vuelta" / roundtrip).

from etl.exporter import export_to_parquet


def test_roundtrip_parquet(spark, tmp_path):
    """Lo que se escribe en Parquet se puede leer igual (mismas filas)."""
    # Arrange: un DataFrame pequeno conocido
    df = spark.createDataFrame(
        [("Hotel A", "CA", 5.0, 1), ("Hotel B", "ME", 3.0, 0)],
        ["hotel_name", "province", "rating", "sentiment"],
    )
    salida = str(tmp_path / "out.parquet")
    # Act: exportamos y volvemos a leer
    export_to_parquet(df, salida)
    leido = spark.read.parquet(salida)
    # Assert: mismas filas y mismas columnas
    assert leido.count() == 2
    assert set(leido.columns) == {"hotel_name", "province", "rating", "sentiment"}


def test_overwrite_no_duplica(spark, tmp_path):
    """Exportar dos veces al mismo path (mode overwrite) no acumula filas."""
    # Arrange
    df = spark.createDataFrame([("Hotel A", "CA", 5.0, 1)],
                               ["hotel_name", "province", "rating", "sentiment"])
    salida = str(tmp_path / "out.parquet")
    # Act: exportamos dos veces
    export_to_parquet(df, salida)
    export_to_parquet(df, salida)
    leido = spark.read.parquet(salida)
    # Assert: sigue habiendo 1 fila (overwrite, no append)
    assert leido.count() == 1
