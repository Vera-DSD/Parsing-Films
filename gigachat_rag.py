import os

# Получаем ключ из переменной окружения (HF Spaces Secrets)
GIGACHAT_API_KEY = os.getenv("GIGACHAT_API_KEY")
if not GIGACHAT_API_KEY:
    raise ValueError("Требуется GIGACHAT_API_KEY в Secrets")