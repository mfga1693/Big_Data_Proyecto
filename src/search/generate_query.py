from sentence_transformers import SentenceTransformer
import json

def main():
    print("Cargando modelo en memoria (esto tomará unos segundos)...")
    modelo = SentenceTransformer("all-MiniLM-L6-v2")
    print("Modelo listo. Puedes empezar a escribir tus búsquedas.\n")
    

    while True:

        pregunta = input("Escribe tu búsqueda (o 'salir' para terminar): ")
        
        if pregunta.lower() == 'salir':
            print("Saliendo...")
            break
            
        if not pregunta.strip():
            print("Por favor, escribe algo.\n")
            continue
            
        print("Traduciendo a vector...")
        vector = modelo.encode(pregunta).tolist()
        
        query_kibana = {
            "knn": {
                "field": "embedding",
                "query_vector": vector,
                "k": 5,
                "num_candidates": 100
            },
            "_source": [
                "hotel_name",
                "review_full_text",
                "rating"
            ]
        }
        
        print("\n" + "="*50)
        print("Copiar todo el bloque siguiente para pegarlo en Kibana Dev Tools:")
        print("="*50)
        print("GET hotel-reviews-semantic/_search")
        print(json.dumps(query_kibana, indent=2))
        print("="*50 + "\n")

if __name__ == "__main__":
    main()