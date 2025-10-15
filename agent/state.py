# /agent/state.py
from typing import TypedDict, List, Dict, Any

class AgentState(TypedDict):
    original_query: str
    is_relevant: bool
    sub_queries: List[str]
    search_results_path: str # Путь к подкаталогу с результатами поиска
    thinking_results_file: str # Путь к файлу с частичными ответами
    final_response: str