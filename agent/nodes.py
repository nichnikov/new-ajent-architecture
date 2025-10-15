import json
import logging
import os
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
import config
import prompts
from tools.search_documents import search_documents
from utils.file_utils import save_json, generate_unique_name, load_json

# Настройка логгирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Инициализация модели
llm = ChatGoogleGenerativeAI(
    model=config.MODEL_NAME,
    temperature=config.LLM_TEMPERATURE,
    google_api_key=config.GEMINI_API_KEY
)

def analyze_relevance(state):
    """Первичный анализ вопроса на релевантность."""
    logging.info("--- УЗЕЛ: АНАЛИЗ РЕЛЕВАНТНОСТИ ---")
    query = state['original_query']
    prompt = PromptTemplate.from_template(prompts.ANALYZE_PROMPT).format(query=query)
    response = llm.invoke(prompt)
    answer = response.content.strip().lower()
    logging.info(f"Ответ модели на релевантность: '{answer}'")
    state['is_relevant'] = "да" in answer
    return state

def generate_subqueries(state):
    """Генерация вспомогательных вопросов."""
    logging.info("--- УЗЕЛ: ГЕНЕРАЦИЯ ПОДЗАПРОСОВ ---")
    query = state['original_query']
    prompt = PromptTemplate.from_template(prompts.SUBQUERY_GEN_PROMPT).format(query=query)
    response = llm.invoke(prompt)
    try:
        sub_queries = json.loads(response.content)
        logging.info(f"Сгенерированы подзапросы: {sub_queries}")
        state['sub_queries'] = sub_queries
    except json.JSONDecodeError:
        logging.error("Не удалось распарсить JSON с подзапросами. Используем основной запрос.")
        state['sub_queries'] = [query]
    return state

def search_and_save(state):
    """Поиск документов и сохранение результатов."""
    logging.info("--- УЗЕЛ: ПОИСК И СОХРАНЕНИЕ ---")
    sub_queries = state['sub_queries']
    dir_name = generate_unique_name(state['original_query'])
    search_dir = os.path.join(config.SEARCH_RESULTS_DIR, dir_name)
    state['search_results_path'] = search_dir

    for query in sub_queries:
        logging.info(f"Выполняется поиск по запросу: '{query}'")
        results = search_documents.invoke({"query": query})
        for doc in results.get("documents", []):
            file_name = generate_unique_name(doc['title']) + ".json"
            file_path = os.path.join(search_dir, file_name)
            # Формат сохранения соответствует ТЗ
            save_data = {
                "query": query,
                "url": doc.get("url"),
                "content": doc.get("content"),
                "title": doc.get("title")
            }
            save_json(save_data, file_path)
            logging.info(f"Результат сохранен в: {file_path}")

    return state

def extract_answers(state):
    """Извлечение ответов из найденных документов."""
    logging.info("--- УЗЕЛ: ИЗВЛЕЧЕНИЕ ОТВЕТОВ ---")
    original_query = state['original_query']
    search_dir = state['search_results_path']
    partial_answers = []

    for filename in os.listdir(search_dir):
        if filename.endswith(".json"):
            filepath = os.path.join(search_dir, filename)
            doc = load_json(filepath)
            content = doc.get("content", "")
            logging.info(f"Анализ файла: {filename} (размер: {len(content)} симв.)")

            # Простая логика обработки больших файлов (можно усложнить)
            if len(content) > config.CHUNK_SIZE:
                # В данном примере просто берем начало, в реальном проекте нужна нарезка на чанки
                content = content[:config.CHUNK_SIZE]

            prompt = PromptTemplate.from_template(prompts.EXTRACT_PROMPT).format(
                query=original_query, fragment=content
            )
            response = llm.invoke(prompt)
            answer = response.content.strip()

            if answer:
                logging.info(f"Найден частичный ответ в '{filename}'")
                partial_answers.append({
                    "answer": answer,
                    "url": doc.get("url"),
                    "fragment": content[:config.MAX_FRAGMENT_SIZE], # Сохраняем только анализируемый фрагмент
                    "file_name": filename,
                    "title": doc.get("title")
                })

    # Сохранение результатов в thinking_results
    file_name = generate_unique_name(original_query) + ".json"
    thinking_file_path = os.path.join(config.THINKING_RESULTS_DIR, file_name)
    save_data = {
        "original_query": original_query,
        "partial_answers": partial_answers
    }
    save_json(save_data, thinking_file_path)
    state['thinking_results_file'] = thinking_file_path
    return state

def audit_answers(state):
    """Аудит и фильтрация ответов."""
    logging.info("--- УЗЕЛ: АУДИТ ОТВЕТОВ ---")
    thinking_file = state['thinking_results_file']
    data = load_json(thinking_file)
    original_query = data['original_query']
    audited_answers = []

    for item in data['partial_answers']:
        prompt = PromptTemplate.from_template(prompts.AUDIT_PROMPT).format(
            query=original_query, fragment=item['fragment'], answer=item['answer']
        )
        response = llm.invoke(prompt)
        try:
            audit_result = json.loads(response.content)
            logging.info(f"Аудит для '{item['title']}': {audit_result}")
            if audit_result.get("fact_check") and audit_result.get("relevance_check"):
                audited_answers.append(item)
                logging.info("--> Ответ прошел аудит.")
            else:
                logging.warning("--> Ответ НЕ прошел аудит и будет отброшен.")
        except json.JSONDecodeError:
            logging.error(f"Не удалось распарсить JSON аудита: {response.content}")

    # Перезаписываем файл только с проверенными ответами
    data['partial_answers'] = audited_answers
    save_json(data, thinking_file)
    logging.info(f"Аудит завершен. Осталось {len(audited_answers)} проверенных ответов.")
    return state

def synthesize_final_response(state):
    """Синтез финального ответа."""
    logging.info("--- УЗЕЛ: СИНТЕЗ ФИНАЛЬНОГО ОТВЕТА ---")
    thinking_file = state['thinking_results_file']
    data = load_json(thinking_file)

    if not data['partial_answers']:
        logging.warning("Нет проверенных ответов для генерации финального ответа.")
        state['final_response'] = "К сожалению, не удалось найти проверенную информацию по вашему вопросу."
        return state

    context_str = "\n\n".join(
        [f"Источник: {item['title']} ({item['url']})\nОтвет из источника: {item['answer']}" for item in data['partial_answers']]
    )

    prompt = PromptTemplate.from_template(prompts.SYNTHESIZE_PROMPT).format(
        query=data['original_query'], context=context_str
    )
    final_response = llm.invoke(prompt)
    state['final_response'] = final_response.content
    logging.info("Финальный ответ сгенерирован.")
    return state

# Узел для случая, если вопрос нерелевантен
def irrelevant_query_response(state):
    logging.info("--- Узел: НЕРЕЛЕВАНТНЫЙ ЗАПРОС ---")
    state['final_response'] = "Ваш вопрос не относится к бухгалтерскому или налоговому учету. Пожалуйста, задайте вопрос по теме."
    return state