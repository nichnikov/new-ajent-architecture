import asyncio
import logging

from pydantic import HttpUrl

from agents.jur_agent.types import SearchResults, UnifiedDoc
from search.action.client import SearchClient
from search.action.parser import DocumentParser
from search.action.schemas import SearchParams

logger = logging.getLogger(__name__)


class ActionSearch:
    name = "internal_search"
    pages = 1

    def __init__(self) -> None:
        self.parser = DocumentParser()

    def search(self, *, search_params: SearchParams, pages: int) -> dict[str, object]:
        return asyncio.run(self.search_async(search_params=search_params, pages=pages))

    async def search_async(self, *, search_params: SearchParams, pages: int) -> dict[str, object]:
        client = SearchClient()

        doc_results = await client.fetch_search_pages_and_docs(
            search_params=search_params,
            pages=pages,
        )
        parsed = []
        errors = []

        for search_result in doc_results:
            if search_result.error is not None:
                item_dict = search_result.item.model_dump()
                errors.append({"item": item_dict, "error": search_result.error})
                logger.error(
                    "doc fetch error | moduleId=%s id=%s | %s",
                    search_result.item.moduleId,
                    search_result.item.id,
                    search_result.error,
                )
                continue
            try:
                search_item = search_result.item
                doc_response = search_result.document

                plain = self.parser.parse(doc_response)

                title = self.parser._clean_text(doc_response["document"]["content"]["title"])

                out = {
                    "id": search_item.id,
                    "moduleId": search_item.moduleId,
                    "url": f"https://1gl.ru/?#/document/{search_item.moduleId}/{search_item.id}",
                    "api_url": search_item.url,
                    "title": title,
                    "plain_text": plain,
                }
                parsed.append(out)

                # лог итогового документа
                title_str = (out["title"] or "") if isinstance(out["title"], str) else ""
                snippet = plain[:800] + ("…" if len(plain) > 800 else "")
                logger.info("DOC #%s/%s | %s\n%s", out["moduleId"], out["id"], title_str, snippet)

            except Exception as e:
                item_dict = search_result.item.model_dump()
                errors.append({"item": item_dict, "error": f"parse error: {e}"})
                logger.exception("parse error | moduleId=%s id=%s", search_result.item.moduleId, search_result.item.id)

        return {
            "items": [result.item.model_dump() for result in doc_results],
            "parsed": parsed,
            "errors": errors,
        }

    def run(self, query: str, size: int) -> SearchResults:
        search_params = SearchParams(
            pubAlias="bss.plus",
            fixedregioncode="77",
            isUseHints="false",
            fstring=query,
            sortby="Relevance",
            status="actual",
            dataformat="json",
            pubdivid=1,
            pubId=220,
        )

        raw_result = self.search(search_params=search_params, pages=self.pages)
        raw = raw_result.get("parsed", [])[:size]

        docs = []
        for r in raw:
            docs.append(
                UnifiedDoc(
                    title=r.get("title", "Без заголовка"),
                    snippet=None,
                    content=r.get("plain_text"),
                    url=HttpUrl(r.get("url", "")),
                    source="internal",
                    doc_type="internal",
                    score_raw=0.0,
                    law_refs=[],  # можно добавить извлечение законов из plain_text
                    hash="",
                )
            )
        return SearchResults(docs=docs, meta={"provider": "internal", "count": str(len(docs))})


# ============================== Пример ==============================

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format="%(levelname)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.DEBUG)
    logging.getLogger("httpcore").setLevel(logging.DEBUG)
    try:
        action_search = ActionSearch()

        test_query = "Как составить и сдать единую (упрощенную) налоговую декларацию"
        results = action_search.run(query=test_query, size=3)

        logging.info(f"\nНайдено документов: {len(results.docs)}")
        logging.info(f"Мета-информация: {results.meta}")

        for i, doc in enumerate(results.docs, 1):
            logging.info(f"\n--- Документ #{i} ---")
            logging.info(f"Заголовок: {doc.title}")
            logging.info(f"URL: {doc.url}")
            logging.info(f"Источник: {doc.source}")
            logging.info(f"Тип: {doc.doc_type}")
            if doc.content:
                logging.info(f"Содержимое: {doc.content[:200]}{'...' if len(doc.content) > 200 else ''}")

    except Exception as e:
        logging.error(f"\nПроизошла ошибка во время теста: {e}")
        import traceback

        traceback.print_exc()
