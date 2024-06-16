from ollama import OllamaClient

def generate_answer_with_ollama(question, context):
    client = OllamaClient()
    response = client.ask(context, question)
    return response['text']
