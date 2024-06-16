import os
from pdf_processing.extract import extract_text_from_pdf
from pdf_processing.chunk import chunk_text
from e_search.save import save_chunks_to_elasticsearch
from e_search.query import ask_question_to_elasticsearch
from llm.generate import generate_answer_with_ollama
import hashlib

def calculate_pdf_hash(pdf_path):
    hasher = hashlib.md5()
    with open(pdf_path, 'rb') as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()

if __name__ == "__main__":
    pdfs_folder = "data"  # Ajuste o caminho se a pasta estiver em outro local
    for filename in os.listdir(pdfs_folder):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(pdfs_folder, filename)
            pdf_hash = calculate_pdf_hash(pdf_path)
            text = extract_text_from_pdf(pdf_path)
            chunks = chunk_text(text)
            save_chunks_to_elasticsearch(chunks, pdf_hash)

    while True:
        question = input("Digite a pergunta: ")
        if question.lower() == 'exit':
            break
        relevant_texts = ask_question_to_elasticsearch(question)
        combined_text = " ".join(relevant_texts)
        answer = generate_answer_with_ollama(question, combined_text)
        print(answer)
