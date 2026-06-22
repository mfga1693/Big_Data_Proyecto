from sentence_transformers import SentenceTransformer
import json


def main():
    print("Cargando modelo en memoria (esto tomará unos segundos)...")
    modelo = SentenceTransformer("all-MiniLM-L6-v2")
    print("Modelo listo. Puedes empezar a escribir tus búsquedas.\n")

    while True:
        pregunta = input("Escribe tu búsqueda (o 'salir' para terminar): ")

        if pregunta.lower() == "salir":
            print("Saliendo...")
            break
        if not pregunta.strip():
            print("Por favor, escribe algo.\n")
            continue

        # ---------- Query SIN vectores (BM25) ----------
        query_bm25 = {
            "query": {
                "match": {"review_full_text": pregunta}
            },
            "size": 5,
            "_source": ["hotel_name", "review_full_text", "rating"]
        }

        # ---------- Query CON vectores (kNN) ----------
        vector = modelo.encode(pregunta).tolist()
        query_knn = {
            "knn": {
                "field": "embedding",
                "query_vector": vector,
                "k": 5,
                "num_candidates": 100
            },
            "_source": ["hotel_name", "review_full_text", "rating"]
        }

        print("\n" + "=" * 60)
        print("1) BM25 (SIN vectores) — pegar en Kibana Dev Tools:")
        print("=" * 60)
        print("GET hotel-reviews-bm25/_search")
        print(json.dumps(query_bm25, indent=2, ensure_ascii=False))

        print("\n" + "=" * 60)
        print("2) kNN (CON vectores) — pegar en Kibana Dev Tools:")
        print("=" * 60)
        print("GET hotel-reviews-semantic/_search")
        print(json.dumps(query_knn, indent=2, ensure_ascii=False))
        print("=" * 60 + "\n")

if __name__ == "__main__":
    main()