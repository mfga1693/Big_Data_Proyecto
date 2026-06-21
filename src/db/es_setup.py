from elasticsearch import Elasticsearch

def setup_elasticsearch():
    es = Elasticsearch("http://elasticsearch:9200")

    bm25_mapping = {
        "mappings": {
            "properties": {
                "hotel_name": {"type": "keyword"},
                "city": {"type": "keyword"},
                "province": {"type": "keyword"},
                "rating": {"type": "float"},
                "review_full_text": {"type": "text"}, # Campo clave para BM25
                "sentiment": {"type": "integer"},
                "region": {"type": "keyword"},
                "division": {"type": "keyword"}
            }
        }
    }
    semantic_mapping = {
        "mappings": {
            "properties": {
                "hotel_name": {"type": "keyword"},
                "city": {"type": "keyword"},
                "province": {"type": "keyword"},
                "rating": {"type": "float"},
                "review_full_text": {"type": "text"},
                "sentiment": {"type": "integer"},
                "region": {"type": "keyword"},
                "division": {"type": "keyword"},
                "embedding": {
                    "type": "dense_vector",
                    "dims": 384,
                    "index": True,
                    "similarity": "cosine"
                }
            }
        }
    }

    index_bm25 = "hotel-reviews-bm25"
    index_semantic = "hotel-reviews-semantic"

    if es.indices.exists(index=index_bm25):
        es.indices.delete(index=index_bm25)
    es.indices.create(index=index_bm25, body=bm25_mapping)
    print(f"Índice '{index_bm25}' creado exitosamente.")

    if es.indices.exists(index=index_semantic):
        es.indices.delete(index=index_semantic)
    es.indices.create(index=index_semantic, body=semantic_mapping)
    print(f"Índice '{index_semantic}' creado exitosamente.")

if __name__ == "__main__":
    setup_elasticsearch()