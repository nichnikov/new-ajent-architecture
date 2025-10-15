import asyncio
import logging
import os
import re
from typing import Any

import aiohttp
import trafilatura
from bs4 import BeautifulSoup
from pydantic import HttpUrl

from agents.jur_agent.types import SearchResults, UnifiedDoc
from search.yandex_search_api import YandexSearchAPIClient
from search.yandex_search_api.client import SearchType


def _get_proxy_config() -> str | None:
    for proxy_var in ["HTTP_PROXY", "HTTPS_PROXY"]:
        proxy_url = os.getenv(proxy_var)
        if proxy_url:
            return proxy_url
    return None


def normalize_whitespace(s: str) -> str:
    """Normalize whitespace in text."""

    return re.sub(r"\s+", " ", (s or "").strip())


def _extract_title_from_html(html: str) -> str:
    """Пытается извлечь заголовок из <title>, либо из og:title / twitter:title."""
    try:
        soup = BeautifulSoup(html or "", "html.parser")
        if soup.title and soup.title.string:
            t = soup.title.string.strip()
            if t:
                return t
        og = soup.find("meta", property="og:title")
        if og and og.get("content"):
            return og["content"].strip()
        tw = soup.find("meta", attrs={"name": "twitter:title"})
        if tw and tw.get("content"):
            return tw["content"].strip()
    except Exception:
        pass
    return "Без заголовка"


def _infer_doc_type(r: dict) -> str:
    url = r.get("url", "")
    if "consultant.ru" in url or "garant.ru" in url:
        return "law"
    if "nalog.gov.ru" in url or "minfin.gov.ru" in url:
        return "gov_letter"
    if "sudact.ru" in url:
        return "court"
    if "forum" in url:
        return "forum"
    if "news" in url:
        return "news"
    return "article"


def _extract_law_refs(r: dict):
    text = " ".join([r.get("title", ""), r.get("snippet", ""), r.get("content", "")])
    refs = []
    for m in re.finditer(r"(НК РФ|ПБУ|ФСБУ|ТК РФ|ГК РФ)(?:[^.,;:]{0,40})?(ст\.?\s?\d+)?", text, flags=re.IGNORECASE):
        code = m.group(1).upper()
        article = m.group(2)
        refs.append({"code": code, "article": article})
    return refs


class YandexSearch:
    def __init__(self):
        oauth_token = os.getenv("YANDEX_OAUTH_TOKEN")
        folder_id = os.getenv("YANDEX_FOLDER_ID")
        self.client = YandexSearchAPIClient(folder_id=folder_id, oauth_token=oauth_token)

    async def _scrape_page(
        self,
        url: str,
        session: aiohttp.ClientSession | None = None,
        proxy: str | None = None,
    ) -> dict[str, str]:
        """
        Приватный метод для загрузки страницы, извлечения заголовка и текста.

        Args:
            url: URL-адрес для скрапинга.
            session: Необязательная общая aiohttp-сессия (для батч-скрапинга).

        Returns:
            Словарь с 'title' и 'content' страницы.
        """
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/134.0.0.0 Safari/537.36 Edg/134.0.0.0"
        }

        own_session = session is None
        if own_session:
            session = aiohttp.ClientSession(headers=headers)
        assert session is not None

        try:
            async with session.get(url, proxy=proxy, timeout=aiohttp.ClientTimeout(total=30)) as r:
                if r.status != 200:
                    return {"title": "Ошибка", "content": f"HTTP статус: {r.status}"}
                # Важно: используем .text(errors="ignore"), чтобы не падать на кривой кодировке
                html = await r.text(errors="ignore")
        except asyncio.TimeoutError:
            return {"title": "Ошибка", "content": f"Тайм-аут при загрузке {url}"}
        except aiohttp.ClientError as e:
            return {"title": "Ошибка", "content": f"Ошибка aiohttp при загрузке {url}: {e}"}
        except Exception as e:
            return {"title": "Ошибка", "content": f"Неожиданная ошибка при загрузке {url}: {e}"}
        finally:
            if own_session:
                await session.close()

        # Заголовок стараемся извлечь заранее
        title = _extract_title_from_html(html)

        # Попробуем извлечь текст через trafilatura
        try:
            extracted = trafilatura.extract(html, include_comments=False, include_tables=False) or ""
        except Exception:
            extracted = None

        if extracted:
            return {"title": title, "content": normalize_whitespace(extracted)}

        # Если trafilatura не смог — fallback через BeautifulSoup
        try:
            soup = BeautifulSoup(html, "html.parser")

            # Удаляем шум
            for tag in soup(["script", "style", "header", "footer", "nav", "aside"]):
                tag.decompose()

            # Извлекаем чистый текст
            content = soup.get_text(separator="\n", strip=True)
            content = normalize_whitespace(content)

            # Если title пустой, попробуем еще раз из soup
            if (not title or title == "Без заголовка") and soup.title and soup.title.string:
                title = soup.title.string.strip() or "Без заголовка"

            return {"title": title, "content": content}
        except Exception as e:
            return {"title": "Ошибка", "content": f"Ошибка при обработке HTML: {e}"}

    def _format_results(self, results: list[dict[str, Any]]) -> str:
        """
        Форматирует обработанные результаты поиска в единую строку для LLM.
        """
        if not results:
            return "Поиск в Yandex не дал результатов."

        formatted_string = ""
        for i, item in enumerate(results, 1):
            title = item.get("title", "Без заголовка")
            url = item.get("url", "Ссылка отсутствует")
            content = item.get("content", "Содержимое отсутствует")

            formatted_string += f"Источник #{i}:\n"
            formatted_string += f"  Название: {title}\n"
            formatted_string += f"  Ссылка: {url}\n"
            formatted_string += f'  Содержимое:\n"""\n{content}\n"""\n\n'

        return formatted_string

    async def search(self, query: str, **kwargs: Any) -> list[dict[str, Any]]:
        """
        Выполняет поиск в Yandex, асинхронно скрапит страницы и возвращает структурированный результат.

        Args:
            query: поисковый запрос
            num_results: (kwarg) количество результатов (по умолчанию 5)
            search_type: (kwarg) тип поиска (по умолчанию SearchType.RUSSIAN)

        Returns:
            Список словарей {title, url, content}.
        """
        num_results = kwargs.get("num_results", 5)
        search_type = kwargs.get("search_type", SearchType.RUSSIAN)
        try:
            links = self.client.get_links(query_text=query, search_type=search_type, n_links=num_results)

            if not links:
                print("Поиск в Yandex не вернул ссылок.")
                return []

            # Параллельный асинхронный скрапинг
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/134.0.0.0 Safari/537.36 Edg/134.0.0.0"
            }
            # Получаем настройки прокси один раз для всех запросов
            proxy_url = _get_proxy_config()
            async with aiohttp.ClientSession(headers=headers) as session:
                tasks = [self._scrape_page(link, session=session, proxy=proxy_url) for link in links if link]
                pages = await asyncio.gather(*tasks, return_exceptions=True)

            processed_items: list[dict[str, Any]] = []
            for link, page_data in zip([l for l in links if l], pages):
                if isinstance(page_data, Exception):
                    logging.error(f"Ошибка при скрапинге {link}: {page_data}")
                    continue
                processed_items.append(
                    {
                        "title": page_data.get("title", "Без заголовка"),
                        "url": link,
                        "content": page_data.get("content", ""),
                    }
                )

            return processed_items

        except Exception as e:
            logging.error(f"Произошла общая ошибка при поиске в Yandex: {e}")
            return []

    def run(self, query: str, size: int) -> SearchResults:
        raw = asyncio.run(self.search(query, num_results=size))
        docs: list[UnifiedDoc] = []
        for r in raw:
            docs.append(
                UnifiedDoc(
                    title=r["title"],
                    snippet=r.get("snippet"),
                    content=r.get("content"),
                    url=HttpUrl(r["url"]),
                    source="yandex",
                    doc_type=_infer_doc_type(r),
                    score_raw=float(r.get("score", 0.0)),
                    law_refs=_extract_law_refs(r),
                    hash="",  # авто-вычислится
                )
            )
        return SearchResults(docs=docs, meta={"provider": "yandex", "count": str(len(docs))})


# ============================== Пример ==============================
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format="%(levelname)s: %(message)s")

    try:
        yandex_search = YandexSearch()

        test_query = "когда сдавать баланс за 2024 год в ГИР БО"
        results = yandex_search.run(query=test_query, size=3)

        print(f"\nНайдено документов: {len(results.docs)}")
        print(f"Мета-информация: {results.meta}")

        for i, doc in enumerate(results.docs, 1):
            print(f"\n--- Документ #{i} ---")
            print(f"Заголовок: {doc.title}")
            print(f"URL: {doc.url}")
            print(f"Источник: {doc.source}")
            print(f"Тип: {doc.doc_type}")
            print(f"Оценка: {doc.score_raw}")
            if doc.law_refs:
                print(f"Ссылки на законы: {doc.law_refs}")
            if doc.content:
                print(f"Содержимое: {doc.content[:200]}{'...' if len(doc.content) > 200 else ''}")

    except ValueError as e:
        print(f"\nОшибка при создании экземпляра: {e}")
    except Exception as e:
        print(f"\nПроизошла ошибка во время теста: {e}")
        import traceback

        traceback.print_exc()
