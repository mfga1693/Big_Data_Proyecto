# test_cleaner.py
#
# Pruebas unitarias del modulo cleaner (funcion mrmusculo): limpieza, join con
# states, columnas derivadas (review_full_text, sentiment) y filtro de idioma.
#
# Patron de cada prueba: AAA (Arrange -> Act -> Assert).
# La fixture `spark` viene de conftest.py (una SparkSession compartida).
#
# El cleaner filtra por idioma con langdetect; si no esta instalado, se saltan.
import pytest

pytest.importorskip("langdetect")

from etl.cleaner import mrmusculo  # noqa: E402

# Nombres de columnas CRUDAS que espera mrmusculo (igual que en el CSV original).
HOTELS_COLS = [
    "name", "city", "province", "reviews.date", "reviews.rating",
    "reviews.text", "reviews.title", "reviews.username",
]
# Columnas del dataset de estados (el join usa "State Code").
STATES_COLS = ["State", "State Code", "Region", "Division"]

# Textos en INGLES claros (para que pasen el filtro de idioma).
EN_TITLE = "Great stay"
EN_TEXT = "The room was very clean and the staff were friendly"
EN_BAD_TITLE = "Bad experience"
EN_BAD_TEXT = "The room was dirty and the service was very slow"


def _states(spark):
    """Helper: crea un DataFrame de estados con CA y ME (para los joins)."""
    return spark.createDataFrame(
        [
            ("California", "CA", "West", "Pacific"),
            ("Maine", "ME", "Northeast", "New England"),
        ],
        STATES_COLS,
    )


def test_crea_review_full_text_y_sentiment(spark):
    """review_full_text junta titulo + texto, y sentiment=1 si rating >= 4."""
    # Arrange: una resena en ingles con rating 5 en provincia CA
    hotels = spark.createDataFrame(
        [("Hotel Sol", "Los Angeles", "CA", "2021-01-01", 5.0,
          EN_TEXT, EN_TITLE, "ana")],
        HOTELS_COLS,
    )
    # Act
    result = mrmusculo(hotels, _states(spark)).collect()
    # Assert
    assert len(result) == 1
    fila = result[0]
    assert fila["review_full_text"] == f"{EN_TITLE} {EN_TEXT}"  # titulo + " " + texto
    assert fila["sentiment"] == 1                                # rating 5 -> positivo


def test_sentiment_negativo_si_rating_bajo(spark):
    """sentiment=0 cuando el rating es menor a 4."""
    # Arrange: rating 3 -> deberia ser negativo
    hotels = spark.createDataFrame(
        [("Hotel Luna", "Portland", "ME", "2021-02-02", 3.0,
          EN_BAD_TEXT, EN_BAD_TITLE, "beto")],
        HOTELS_COLS,
    )
    # Act
    result = mrmusculo(hotels, _states(spark)).collect()
    # Assert
    assert len(result) == 1
    assert result[0]["sentiment"] == 0


def test_concat_ws_sin_titulo_no_deja_nulo(spark):
    """Si el titulo es nulo, review_full_text queda igual al texto (no nulo)."""
    # Arrange: una fila CON titulo (para que Spark infiera el tipo) y otra SIN titulo
    hotels = spark.createDataFrame(
        [
            ("Hotel CT", "LA", "CA", "2021-01-01", 5.0, EN_TEXT, EN_TITLE, "u0"),
            ("Hotel NT", "LA", "CA", "2021-01-01", 5.0, EN_BAD_TEXT, None, "u1"),
        ],
        HOTELS_COLS,
    )
    # Act
    result = mrmusculo(hotels, _states(spark)).collect()
    # Assert: la fila sin titulo tiene review_full_text = solo el texto (no nulo)
    sin_titulo = [r for r in result if r["hotel_name"] == "Hotel NT"][0]
    assert sin_titulo["review_full_text"] == EN_BAD_TEXT


def test_filtra_ratings_invalidos(spark):
    """Solo se conservan ratings entre 1 y 5; los demas se descartan."""
    # Arrange: una valida (5) y una invalida (9)
    hotels = spark.createDataFrame(
        [
            ("Hotel A", "LA", "CA", "2021-01-01", 5.0, EN_TEXT, EN_TITLE, "u1"),
            ("Hotel B", "LA", "CA", "2021-01-01", 9.0, EN_TEXT, EN_TITLE, "u2"),
        ],
        HOTELS_COLS,
    )
    # Act
    result = mrmusculo(hotels, _states(spark)).collect()
    # Assert: solo queda la del rating valido
    assert len(result) == 1
    assert result[0]["rating"] == 5.0


def test_elimina_resenas_con_texto_nulo(spark):
    """Las filas con review_text nulo se eliminan."""
    # Arrange: la primera tiene texto, la segunda es nula
    hotels = spark.createDataFrame(
        [
            ("Hotel A", "LA", "CA", "2021-01-01", 4.0, EN_TEXT, EN_TITLE, "u1"),
            ("Hotel B", "LA", "CA", "2021-01-01", 4.0, None, EN_TITLE, "u2"),
        ],
        HOTELS_COLS,
    )
    # Act
    result = mrmusculo(hotels, _states(spark)).collect()
    # Assert: solo sobrevive la fila con texto
    assert len(result) == 1
    assert result[0]["hotel_name"] == "Hotel A"


def test_filtra_resenas_no_inglesas(spark):
    """Las resenas que no estan en ingles se descartan (filtro de idioma)."""
    # Arrange: una en ingles y una en aleman
    hotels = spark.createDataFrame(
        [
            ("Hotel EN", "LA", "CA", "2021-01-01", 5.0, EN_TEXT, EN_TITLE, "u1"),
            ("Hotel DE", "LA", "CA", "2021-01-01", 5.0,
             "Das Zimmer war sehr schmutzig und das Personal war unfreundlich",
             "Schlechte Erfahrung", "u2"),
        ],
        HOTELS_COLS,
    )
    # Act
    result = mrmusculo(hotels, _states(spark)).collect()
    # Assert: solo queda la reseña en ingles
    assert len(result) == 1
    assert result[0]["hotel_name"] == "Hotel EN"


def test_join_inner_descarta_provincias_sin_estado(spark):
    """El inner join descarta resenas cuya provincia no existe en states."""
    # Arrange: CA existe en states; ZZ no
    hotels = spark.createDataFrame(
        [
            ("Hotel A", "LA", "CA", "2021-01-01", 5.0, EN_TEXT, EN_TITLE, "u1"),
            ("Hotel Z", "??", "ZZ", "2021-01-01", 5.0, EN_TEXT, EN_TITLE, "u2"),
        ],
        HOTELS_COLS,
    )
    # Act
    result = mrmusculo(hotels, _states(spark)).collect()
    # Assert: solo queda la provincia que cruzo (CA)
    assert len(result) == 1
    assert result[0]["province"] == "CA"


def test_columnas_de_salida(spark):
    """El DataFrame final tiene exactamente las 9 columnas esperadas."""
    # Arrange
    hotels = spark.createDataFrame(
        [("Hotel A", "LA", "CA", "2021-01-01", 5.0, EN_TEXT, EN_TITLE, "u1")],
        HOTELS_COLS,
    )
    # Act
    result = mrmusculo(hotels, _states(spark))
    # Assert: nombres y orden de columnas
    assert result.columns == [
        "hotel_name", "city", "province", "review_date",
        "rating", "review_full_text", "sentiment", "region", "division",
    ]


def test_conserva_region_y_division(spark):
    """El join con states aporta region y division al resultado final."""
    # Arrange: provincia CA -> West / Pacific (segun el helper _states)
    hotels = spark.createDataFrame(
        [("Hotel A", "LA", "CA", "2021-01-01", 5.0, EN_TEXT, EN_TITLE, "u1")],
        HOTELS_COLS,
    )
    # Act
    fila = mrmusculo(hotels, _states(spark)).collect()[0]
    # Assert
    assert fila["region"] == "West"
    assert fila["division"] == "Pacific"
