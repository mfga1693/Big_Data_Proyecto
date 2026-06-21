# test_cleaner.py
# Autor: Andres
#
# Pruebas unitarias del modulo cleaner (funcion mrmusculo): limpieza, join con
# states y columnas derivadas review_full_text y sentiment.
#
# Patron de cada prueba: AAA (Arrange -> Act -> Assert).
# La fixture `spark` viene de conftest.py (una SparkSession compartida).

from etl.cleaner import mrmusculo

# Nombres de columnas CRUDAS que espera mrmusculo (igual que en el CSV original).
HOTELS_COLS = [
    "name", "city", "province", "reviews.date", "reviews.rating",
    "reviews.text", "reviews.title", "reviews.username",
]
# Columnas del dataset de estados (el join usa "State Code").
STATES_COLS = ["State", "State Code", "Region", "Division"]


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
    # Arrange: una resena con rating 5 en provincia CA
    hotels = spark.createDataFrame(
        [("Hotel Sol", "Los Angeles", "CA", "2021-01-01", 5.0,
          "Clean room", "Great", "ana")],
        HOTELS_COLS,
    )
    # Act
    result = mrmusculo(hotels, _states(spark)).collect()
    # Assert
    assert len(result) == 1
    fila = result[0]
    assert fila["review_full_text"] == "Great Clean room"  # titulo + " " + texto
    assert fila["sentiment"] == 1                            # rating 5 -> positivo


def test_sentiment_negativo_si_rating_bajo(spark):
    """sentiment=0 cuando el rating es menor a 4."""
    # Arrange: rating 3 -> deberia ser negativo
    hotels = spark.createDataFrame(
        [("Hotel Luna", "Portland", "ME", "2021-02-02", 3.0,
          "Noisy", "Bad", "beto")],
        HOTELS_COLS,
    )
    # Act
    result = mrmusculo(hotels, _states(spark)).collect()
    # Assert
    assert len(result) == 1
    assert result[0]["sentiment"] == 0


def test_filtra_ratings_invalidos(spark):
    """Solo se conservan ratings entre 1 y 5; los demas se descartan."""
    # Arrange: una valida (5) y una invalida (9)
    hotels = spark.createDataFrame(
        [
            ("Hotel A", "LA", "CA", "2021-01-01", 5.0, "ok", "t1", "u1"),
            ("Hotel B", "LA", "CA", "2021-01-01", 9.0, "raro", "t2", "u2"),
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
            ("Hotel A", "LA", "CA", "2021-01-01", 4.0, "buen texto", "t1", "u1"),
            ("Hotel B", "LA", "CA", "2021-01-01", 4.0, None, "t2", "u2"),
        ],
        HOTELS_COLS,
    )
    # Act
    result = mrmusculo(hotels, _states(spark)).collect()
    # Assert: solo sobrevive la fila con texto
    assert len(result) == 1
    assert result[0]["hotel_name"] == "Hotel A"


def test_join_inner_descarta_provincias_sin_estado(spark):
    """El inner join descarta resenas cuya provincia no existe en states."""
    # Arrange: CA existe en states; ZZ no
    hotels = spark.createDataFrame(
        [
            ("Hotel A", "LA", "CA", "2021-01-01", 5.0, "ok", "t1", "u1"),
            ("Hotel Z", "??", "ZZ", "2021-01-01", 5.0, "ok", "t2", "u2"),
        ],
        HOTELS_COLS,
    )
    # Act
    result = mrmusculo(hotels, _states(spark)).collect()
    # Assert: solo queda la provincia que cruzo (CA)
    assert len(result) == 1
    assert result[0]["province"] == "CA"


def test_columnas_de_salida(spark):
    """El DataFrame final tiene exactamente las 7 columnas esperadas."""
    # Arrange
    hotels = spark.createDataFrame(
        [("Hotel A", "LA", "CA", "2021-01-01", 5.0, "ok", "t1", "u1")],
        HOTELS_COLS,
    )
    # Act
    result = mrmusculo(hotels, _states(spark))
    # Assert: nombres y orden de columnas
    assert result.columns == [
        "hotel_name", "city", "province", "review_date",
        "rating", "review_full_text", "sentiment",
    ]
