import os
from dotenv import load_dotenv

load_dotenv()

# Загружаем API ключ из .env файла
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("Не найден GEMINI_API_KEY. Пожалуйста, создайте файл .env и добавьте его туда.")

# Настройки модели
# Важно: На данный момент модель "gemini-2.5-pro" является вымышленной.
# Используйте актуальную доступную модель, например, "gemini-1.5-pro-latest"
MODEL_NAME = "gemini-1.5-pro-latest"
LLM_TEMPERATURE = 0.1 # Низкая температура для более предсказуемых и фактических ответов

# Настройки для обработки документов
CHUNK_SIZE = 8000  # Размер части для чтения больших документов (в символах)
MAX_FRAGMENT_SIZE = 10000 # Максимальный размер фрагмента для анализа (в символах)

# Пути к директориям
SEARCH_RESULTS_DIR = "search_results"
THINKING_RESULTS_DIR = "thinking_results"