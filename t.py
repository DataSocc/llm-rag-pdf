import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader

# Configuração do modelo
model_name = 'sentence-transformers/all-MiniLM-L6-v2'
model = SentenceTransformer(model_name)

# Função para extrair texto de PDFs
def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as f:
        pdf = PdfReader(f)
        text = ''
        for page_num in range(len(pdf.pages)):
            page = pdf.pages[page_num]
            text += page.extract_text()
    return text

# Função para dividir o texto em chunks menores
def chunk_text(text, chunk_size=100):
    words = text.split()
    chunks = [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
    return chunks

# Indexação e encoding de PDFs em chunks menores
def index_pdfs(pdf_dir, faiss_index, known_pdfs):
    for pdf_file in os.listdir(pdf_dir):
        if pdf_file.endswith('.pdf') and pdf_file not in known_pdfs:
            pdf_path = os.path.join(pdf_dir, pdf_file)
            text = extract_text_from_pdf(pdf_path)
            chunks = chunk_text(text)
            embeddings = model.encode(chunks)
            faiss_index.add(np.array(embeddings, dtype=np.float32))
            known_pdfs[pdf_file] = chunks

# Armazenamento em memória
dimension = model.get_sentence_embedding_dimension()
faiss_index = faiss.IndexFlatL2(dimension)
known_pdfs = {}

# Indexando PDFs locais
pdf_directory = 'data'
index_pdfs(pdf_directory, faiss_index, known_pdfs)

# Função de busca
def search_query(query, faiss_index, top_k=5):
    query_embedding = model.encode([query])
    D, I = faiss_index.search(np.array(query_embedding, dtype=np.float32), top_k)
    return I

# Função para decodificar embeddings de volta para texto
def decode_embeddings(indices, known_pdfs):
    results = []
    for index in indices[0]:
        if index != -1:
            for pdf_file, chunks in known_pdfs.items():
                if index < len(chunks):
                    results.append(chunks[index])
                    break
                else:
                    index -= len(chunks)
    return results

# Função para processar a pergunta do usuário e fornecer a resposta
def answer_question(query, faiss_index, known_pdfs, top_k=5):
    indices = search_query(query, faiss_index, top_k)
    result_texts = decode_embeddings(indices, known_pdfs)
    if result_texts:
        print(f"\nPergunta: {query}")
        print("Resposta:")
        for i, text in enumerate(result_texts):
            print(f"{i+1}. {text[:5000]}...")  # Limite de 500 caracteres por trecho
    else:
        print("Nenhuma resposta encontrada.")

# Loop contínuo para receber perguntas do usuário
while True:
    query = input("\nDigite sua pergunta (ou 'sair' para terminar): ")
    if query.lower() == 'sair':
        break
    answer_question(query, faiss_index, known_pdfs)

print("Sessão encerrada.")
