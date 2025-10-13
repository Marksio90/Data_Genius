from openai import OpenAI
from .settings import settings

# Klient OpenAI będzie inicjalizowany dynamicznie z session state
client = None
MODEL = settings.openai_model

def get_openai_client(api_key: str):
    """Tworzy klienta OpenAI z podanym kluczem API"""
    if api_key and api_key:
        return OpenAI(api_key=api_key)
    return None