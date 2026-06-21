# test_loader.py
#
# Pruebas unitarias del modulo loader (funcion load_data): que lea correctamente
# los dos CSV (hoteles y estados) y devuelva DataFrames con las columnas y filas
# esperadas.
#
# Usamos la fixture `tmp_path` de pytest: una carpeta temporal y limpia para cada
# test, donde escribimos CSV de prueba (asi no dependemos de los datos reales).

from etl.loader import load_data

# Contenido de un CSV de hoteles minimo (con los nombres de columna del original).
HOTELS_CSV = (
    "name,city,province,reviews.date,reviews.rating,reviews.text,reviews.title,reviews.username\n"
    "Hotel A,LA,CA,2021-01-01,5,Great stay,Good,ana\n"
    "Hotel B,Portland,ME,2021-02-02,3,Noisy,Bad,beto\n"
)
STATES_CSV = (
    "State,State Code,Region,Division\n"
    "California,CA,West,Pacific\n"
    "Maine,ME,Northeast,New England\n"
)


def _escribir(tmp_path, nombre, contenido):
    """Helper: escribe un CSV temporal y devuelve su ruta como texto."""
    ruta = tmp_path / nombre
    ruta.write_text(contenido)
    return str(ruta)


def test_carga_filas_y_columnas(spark, tmp_path):
    """load_data lee ambos CSV con la cantidad de filas correcta."""
    # Arrange: escribimos los dos CSV temporales
    hotels_path = _escribir(tmp_path, "hotels.csv", HOTELS_CSV)
    states_path = _escribir(tmp_path, "states.csv", STATES_CSV)
    # Act
    hotels_df, states_df = load_data(hotels_path, states_path)
    # Assert: cada uno con sus 2 filas
    assert hotels_df.count() == 2
    assert states_df.count() == 2


def test_columnas_clave_presentes(spark, tmp_path):
    """Las columnas que el resto del ETL necesita estan presentes tras la carga."""
    # Arrange
    hotels_path = _escribir(tmp_path, "hotels.csv", HOTELS_CSV)
    states_path = _escribir(tmp_path, "states.csv", STATES_CSV)
    # Act
    hotels_df, states_df = load_data(hotels_path, states_path)
    # Assert: columnas crudas que usa cleaner, y la llave del join en states
    assert "name" in hotels_df.columns
    assert "reviews.rating" in hotels_df.columns
    assert "State Code" in states_df.columns
