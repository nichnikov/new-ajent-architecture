# main.py
import logging
from agent.graph import create_agent_graph

# Настройка логгирования, чтобы видеть процесс работы агента
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    # Создаем и компилируем граф
    app = create_agent_graph()

    # --- Примеры вопросов для тестирования ---

    # 1. Релевантный вопрос, который запустит полный цикл
    user_query_relevant = "Правила ведения кассовых операций для ИП в 2025 году"

    # 2. Нерелевантный вопрос, который должен быть сразу отфильтрован
    user_query_irrelevant = "Какой рецепт у борща?"

    # ----------------------------------------

    print("="*50)
    print(f"ТЕСТ 1: РЕЛЕВАНТНЫЙ ЗАПРОС")
    print(f"Входящий вопрос: {user_query_relevant}")
    print("="*50)

    # Запускаем граф с релевантным вопросом
    inputs = {"original_query": user_query_relevant}
    final_state = app.invoke(inputs)

    # Выводим финальный ответ
    print("\n" + "="*50)
    print("ИТОГОВЫЙ ОТВЕТ АГЕНТА:")
    print(final_state['final_response'])
    print("="*50 + "\n")


    print("="*50)
    print(f"ТЕСТ 2: НЕРЕЛЕВАНТНЫЙ ЗАПРОС")
    print(f"Входящий вопрос: {user_query_irrelevant}")
    print("="*50)

    # Запускаем граф с нерелевантным вопросом
    inputs_irrelevant = {"original_query": user_query_irrelevant}
    final_state_irrelevant = app.invoke(inputs_irrelevant)

    # Выводим финальный ответ
    print("\n" + "="*50)
    print("ИТОГОВЫЙ ОТВЕТ АГЕНТА:")
    print(final_state_irrelevant['final_response'])
    print("="*50)


if __name__ == "__main__":
    main()