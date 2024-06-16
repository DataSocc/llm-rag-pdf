from sentence_transformers import SentenceTransformer
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document

# Carregar o modelo SBERT mais preciso
model_name = 'all-mpnet-base-v2'
model = SentenceTransformer(model_name)

# Criar a classe de embeddings do Hugging Face
class SBERTEmbeddings(HuggingFaceEmbeddings):
    def embed_documents(self, documents):
        texts = [doc.page_content for doc in documents]
        return model.encode(texts, convert_to_tensor=True).tolist()

# Lista de sentenças para as quais você deseja gerar embeddings
sentences = [
    "Este é um exemplo de sentença.",
    "O modelo SBERT é ótimo para gerar embeddings.",
    "As embeddings são representações vetoriais de texto."
]

# Criar documentos para FAISS
documents = [Document(page_content=sentence) for sentence in sentences]

# Inicializar a classe de embeddings
embeddings = SBERTEmbeddings(model_name=model_name)

# Gerar embeddings para os documentos
sentence_embeddings = embeddings.embed_documents(documents)

# Criar um Vector Store com FAISS
vector_store = FAISS.from_documents(documents, embeddings)

# Função para realizar uma busca com uma pergunta e encontrar o trecho mais próximo
def find_closest_sentence(query):
    query_embedding = embeddings.embed_documents([Document(page_content=query)])
    results = vector_store.similarity_search(query, k=1)
    return results[0]
