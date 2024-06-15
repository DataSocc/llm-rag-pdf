from elasticsearch import Elasticsearch
import os

def ask_question_to_elasticsearch(question, index_name="pdf_chunks"):
    es = Elasticsearch([{'host': os.getenv('ELASTICSEARCH_HOST', 'localhost'), 'port': int(os.getenv('ELASTICSEARCH_PORT', 9200))}])
    
    search_body = {
        "query": {
            "match": {
                "text": question
            }
        }
    }
    
    response = es.search(index=index_name, body=search_body)
    hits = response['hits']['hits']
    relevant_texts = [hit['_source']['text'] for hit in hits]
    
    return relevant_texts
