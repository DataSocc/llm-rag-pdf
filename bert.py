from sentence_transformers import SentenceTransformer

# Carregar o modelo SBERT mais preciso
model = SentenceTransformer('all-mpnet-base-v2')

# Lista de sentenças para as quais você deseja gerar embeddings
sentences = [
    "Este é um exemplo de sentença.",
    "O modelo SBERT é ótimo para gerar embeddings.",
    "As embeddings são representações vetoriais de texto."
]

# Gerar embeddings
embeddings = model.encode(sentences)

# Exibir os embeddings
for i, embedding in enumerate(embeddings):
    print(f"Embedding da sentença {i+1}: {embedding}")

# Ver o formato da embedding
print(f"Tamanho da embedding: {len(embeddings[0])}")
