import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Literal

from langchain_core.tools import tool

from config import AppConfig
from tools.searching.search_action import ActionSearch
from tools.searching.search_yandex import YandexSearch
from utils.types import SearchResults

logger = logging.getLogger(__name__)


@tool
def search_documents(
    query: str,
    source: Literal["internal", "yandex", "both"] = "both",
    limit: int = 5,
) -> dict:
    """
    Ищет релевантные документы по бухгалтерии и налогообложению.

    Используй этот инструмент когда нужна информация о:
    - Законах и нормативных актах
    - Правилах ведения бухучёта
    - Налоговых декларациях
    - Судебной практике

    Примеры запросов:
    - "Как заполнить налоговую декларацию УСН за 2024 год"
    - "Статья НК РФ об НДС для IT компаний"
    - "Правила ведения кассовых операций"

    Args:
        query: Чёткий поисковый запрос по юридической или бухгалтерской тематике
        source: Где искать документы - internal (внутренняя база), yandex (интернет), both (оба источника). По умолчанию both
        limit: Максимальное количество результатов от 1 до 10. По умолчанию 5

    Returns:
        Словарь с найденными документами, содержащими title, url, snippet, source, doc_type, score
    """
    logger.info(f"Search documents called: query='{query}', source={source}, limit={limit}")

    # Валидация параметров
    if not query or not query.strip():
        logger.warning("Empty query provided")
        return {
            "documents": [],
            "total_found": 0,
            "query": query,
            "source": source,
            "internal_docs": [],
            "yandex_docs": [],
            "merged_docs": [],
        }

    if limit < 1:
        limit = 1
    elif limit > 10:
        limit = 10

    if source not in ["internal", "yandex", "both"]:
        source = "both"

    cfg = AppConfig()
    yandex = YandexSearch()
    internal = ActionSearch()

    yandex_results = SearchResults()
    internal_results = SearchResults()

    def safe_yandex_search():
        nonlocal yandex_results
        if source in ["yandex", "both"]:
            try:
                yandex_results = yandex.run(query, limit)
                logger.info(f"Yandex search returned {len(yandex_results.docs)} documents")
            except Exception as e:
                logger.error(f"Yandex search error: {e}")

    def safe_internal_search():
        nonlocal internal_results
        if source in ["internal", "both"]:
            try:
                internal_results = internal.run(query, limit)
                logger.info(f"Internal search returned {len(internal_results.docs)} documents")
            except Exception as e:
                logger.error(f"Internal search error: {e}")

    with ThreadPoolExecutor(max_workers=2) as executor:
        yandex_future = executor.submit(safe_yandex_search)
        internal_future = executor.submit(safe_internal_search)
        yandex_future.result()
        internal_future.result()

    all_docs = []
    if source in ["internal", "both"]:
        all_docs.extend(internal_results.docs)
    if source in ["yandex", "both"]:
        all_docs.extend(yandex_results.docs)

    logger.info(f"Total documents before processing: {len(all_docs)}")

    from agents.jur_agent.nodes.normalize_merge import NormalizeMergeRerank

    nmr = NormalizeMergeRerank(cfg=cfg)

    deduped = nmr._dedup(all_docs)
    scored = [nmr._score(d) for d in deduped]
    ranked = sorted(scored, key=lambda d: d.score_rank, reverse=True)[:limit]

    logger.info(f"Returning {len(ranked)} documents after dedup and ranking")

    formatted_docs = []
    for doc in ranked:
        formatted_docs.append(
            {
                "title": doc.title,
                "url": str(doc.url),
                "snippet": doc.snippet or doc.content[:200] if doc.content else "",
                "source": doc.source,
                "doc_type": doc.doc_type,
                "score": round(doc.score_rank, 2),
            }
        )

    result = {
        "documents": formatted_docs,
        "total_found": len(ranked),
        "query": query,
        "source": source,
    }

    return result
