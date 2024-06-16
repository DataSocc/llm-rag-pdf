# Instalação das dependências (execute no terminal)
# !pip install transformers torch pymupdf elasticsearch requests

# Importar as bibliotecas necessárias
import os
import fitz  # PyMuPDF
from typing import List
from elasticsearch import Elasticsearch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Configuração do Elasticsearch com tempo de espera maior
es = Elasticsearch(
    [{"host": "localhost", "port": 9200}],
    timeout=60,  # Tempo de espera aumentado para 60 segundos
    max_retries=10,
    retry_on_timeout=True
)
index_name = "pdf_chunks"

# Carregar o modelo e o tokenizer uma vez
model_name = "EleutherAI/gpt-neo-125M"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Definir pad_token_id como eos_token_id se pad_token não estiver definido
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Função para verificar se um documento já está indexado no Elasticsearch
def is_document_indexed(document_id: str, index_name: str) -> bool:
    try:
        es.get(index=index_name, id=document_id)
        return True
    except:
        return False

# Função para extrair texto do PDF
def extract_text_from_pdf(pdf_path: str) -> str:
    pdf_document = fitz.open(pdf_path)
    text = ""
    for page_num in range(pdf_document.page_count):
        page = pdf_document.load_page(page_num)
        text += page.get_text()
    return text

# Função para dividir o texto em chunks
def chunk_text(text: str, chunk_size: int = 500) -> List[str]:
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

# Função para indexar chunks no Elasticsearch
def index_chunks(chunks: List[str], index_name: str, document_id: str):
    for i, chunk in enumerate(chunks):
        es.index(index=index_name, id=f"{document_id}_{i}", document={"content": chunk})

# Função para gerar resposta usando gpt-neo-125M
def generate_response_with_llm(query: str, context: List[str]) -> str:
    input_text = query + " " + " ".join(context)
    inputs = tokenizer.encode_plus(input_text, return_tensors="pt", max_length=1024, truncation=True, padding="max_length")
    outputs = model.generate(inputs['input_ids'], attention_mask=inputs['attention_mask'], max_new_tokens=500, num_return_sequences=1)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Função para buscar chunks relevantes no Elasticsearch
def search_elasticsearch(query: str, index_name: str, top_k: int = 5) -> List[str]:
    response = es.search(index=index_name, query={
        "match": {
            "content": query
        }
    }, size=top_k)
    hits = response["hits"]["hits"]
    return [hit["_source"]["content"] for hit in hits]

# Função para responder a pergunta usando o LLM e Elasticsearch
def answer_question(query: str) -> str:
    relevant_chunks = search_elasticsearch(query, index_name)
    response = generate_response_with_llm(query, relevant_chunks)
    return response

# Função para processar novos PDFs
def process_new_pdfs(data_dir: str):
    pdf_files = [f for f in os.listdir(data_dir) if f.endswith('.pdf')]
    for pdf_file in pdf_files:
        document_id = os.path.splitext(pdf_file)[0]
        if not is_document_indexed(document_id, index_name):
            pdf_path = os.path.join(data_dir, pdf_file)
            pdf_text = extract_text_from_pdf(pdf_path)
            chunks = chunk_text(pdf_text)
            index_chunks(chunks, index_name, document_id)
            print(f"Indexado: {pdf_file}")
        else:
            print(f"Já indexado: {pdf_file}")

# Processar novos PDFs na pasta 'data'
data_dir = "data"
process_new_pdfs(data_dir)

# Loop interativo para perguntas e respostas
print("Sistema de Perguntas e Respostas. Digite 'sair' para encerrar.")
while True:
    question = input("Faça sua pergunta: ")
    if question.lower() == 'sair':
        break
    answer = answer_question(question)
    print(f"Resposta: {answer}")
