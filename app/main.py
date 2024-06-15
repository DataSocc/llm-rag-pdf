from pdf_processing.extract import extract_text_from_pdf
from pdf_processing.chunk import chunk_text
from elasticsearch.save import save_chunks_to_elasticsearch
from elasticsearch.query import ask_question_to_elasticsearch
from ollama.generate import generate_answer_with_ollama
import hashlib

def calculate_pdf_hash(pdf_path):
    hasher = hashlib.md5()
    with open(pdf_path, 'rb') as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()

if __name__ == "__main__":
    pdf_path = "data/path_to_your_pdf.pdf"
    pdf_hash = calculate_pdf_hash(pdf_path)
    text = extract_text_from_pdf(pdf_path)
    chunks = chunk_text(text)
    save_chunks_to_elasticsearch(chunks, pdf_hash)

    question = "Sua pergunta aqui"
    context = ask_question_to_elasticsearch(question)
    answer = generate_answer_with_ollama(question, context)
    print(answer)
