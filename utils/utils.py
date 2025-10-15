import json
import os
import re
from datetime import datetime
from typing import Any, Dict, List

def save_json(data: Any, filepath: str):
    """Сохраняет данные в JSON файл, создавая директории при необходимости."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def load_json(filepath: str) -> Any:
    """Загружает данные из JSON файла."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def generate_unique_name(query: str) -> str:
    """Генерирует уникальное имя для папки или файла на основе запроса и времени."""
    # Удаляем недопустимые символы и обрезаем до 50
    safe_query = re.sub(r'[\\/*?:"<>|]', "", query)[:50]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{timestamp}_{safe_query.replace(' ', '_')}"