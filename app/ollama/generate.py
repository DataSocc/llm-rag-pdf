from ollama import OllamaClient

def generate_answer_with_ollama(question, context):
    client = OllamaClient()
    combined_context = " ".join(context)
    response = client.ask(combined_context, question)
    return response['text']
