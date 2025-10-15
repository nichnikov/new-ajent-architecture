# /agent/graph.py
from langgraph.graph import StateGraph, END
from .state import AgentState
from . import nodes

def create_agent_graph():
    """Создает и компилирует граф агента."""
    graph = StateGraph(AgentState)

    # Добавление узлов графа
    graph.add_node("analyze_relevance", nodes.analyze_relevance)
    graph.add_node("generate_subqueries", nodes.generate_subqueries)
    graph.add_node("search_and_save", nodes.search_and_save)
    graph.add_node("extract_answers", nodes.extract_answers)
    graph.add_node("audit_answers", nodes.audit_answers)
    graph.add_node("synthesize_final_response", nodes.synthesize_final_response)
    graph.add_node("irrelevant_query_response", nodes.irrelevant_query_response)


    # Определение ребер графа
    graph.set_entry_point("analyze_relevance")

    # Условное ветвление после анализа
    graph.add_conditional_edges(
        "analyze_relevance",
        lambda state: "generate_subqueries" if state["is_relevant"] else "irrelevant_query_response"
    )

    # Основная последовательность действий
    graph.add_edge("generate_subqueries", "search_and_save")
    graph.add_edge("search_and_save", "extract_answers")
    graph.add_edge("extract_answers", "audit_answers")
    graph.add_edge("audit_answers", "synthesize_final_response")

    # Завершение работы
    graph.add_edge("synthesize_final_response", END)
    graph.add_edge("irrelevant_query_response", END)

    # Компиляция графа
    return graph.compile()