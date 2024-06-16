from elasticsearch import Elasticsearch, helpers
import os

def save_chunks_to_elasticsearch(chunks, pdf_hash, index_name="pdf_chunks"):
    es = Elasticsearch([{'host': os.getenv('ELASTICSEARCH_HOST', 'localhost'), 'port': int(os.getenv('ELASTICSEARCH_PORT', 9200))}])
    
    # Check if PDF hash already exists
    if es.exists(index=index_name, id=pdf_hash):
        print("PDF already exists in the index.")
        return

    # Index the PDF hash to mark it as processed
    es.index(index=index_name, id=pdf_hash, body={"processed": True})

    actions = [
        {
            "_index": index_name,
            "_type": "_doc",
            "_id": f"{pdf_hash}_chunk_{idx}",
            "_source": {"text": chunk}
        }
        for idx, chunk in enumerate(chunks)
    ]
    
    helpers.bulk(es, actions)
