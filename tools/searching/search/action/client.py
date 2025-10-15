import asyncio
from collections.abc import Iterable

import httpx

from .schemas import SearchItem, SearchParams, SearchResult

SEARCH_URL = "https://1gl.ru/system/content/search-new/"
DOC_API_URL = "https://site-backend-ss.prod.ss.aservices.tech/api/v1/desktop/document_get-by-id"

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)
HEADERS = {
    "Accept": "application/json, text/plain, */*",
    "User-Agent": USER_AGENT,
    "Referer": "https://1gl.ru/",
    "X-Requested-With": "XMLHttpRequest",
    "Accept-Language": "ru-RU,ru;q=0.9,en;q=0.8",
}


class SearchClient:
    def __init__(self) -> None:
        self.timeout = 15.0
        self.max_connections = 50

    @staticmethod
    def _extract_items(search_page_json: dict[str, object]) -> list[dict[str, object]]:
        """Extract items from search page response."""
        items = search_page_json["data"]["searchResponse"]["items"]
        if not isinstance(items, list):
            return []

        # Validate and convert items to dicts
        validated_items = []
        for item in items:
            validated_items.append(SearchItem.model_validate(item).model_dump())
        return validated_items

    @staticmethod
    def _build_doc_url(base_doc_url: str, module_id: int | str, document_id: int | str) -> str:
        return f"{base_doc_url}?moduleId={module_id}&documentId={document_id}"

    async def _search_pages(
        self,
        *,
        client: httpx.AsyncClient,
        base_search_url: str,
        search_params: SearchParams,
        pages: int,
    ) -> list[dict[str, object]]:
        if pages <= 0:
            return []

        async def fetch_page(p: int) -> dict[str, object]:
            resp = await client.get(base_search_url, params={**search_params.model_dump(exclude_none=True), "page": p})
            resp.raise_for_status()
            ct = resp.headers.get("content-type", "")
            if "application/json" not in ct:
                snippet = resp.text[:300].replace("\n", " ")
                raise httpx.HTTPError(f"Unexpected content-type: {ct}. Snippet: {snippet!r}")

            return resp.json()

        return await asyncio.gather(*[asyncio.create_task(fetch_page(p)) for p in range(1, pages + 1)])

    async def _fetch_docs(
        self,
        *,
        client: httpx.AsyncClient,
        items: Iterable[dict[str, object]],
        base_doc_url: str,
    ) -> list[SearchResult]:
        results = []

        async def fetch_one(item: dict[str, object]) -> None:
            module_id = item.get("moduleId")
            doc_id = item.get("id")

            url = self._build_doc_url(base_doc_url, module_id, doc_id)
            item_with_url = {**dict(item), "url": url}

            search_item = SearchItem.model_validate(item_with_url)

            try:
                resp = await client.get(url)
                resp.raise_for_status()
                ct = resp.headers.get("content-type", "")
                if "application/json" not in ct:
                    snippet = resp.text[:300].replace("\n", " ")
                    raise httpx.HTTPError(f"Unexpected content-type: {ct}. Snippet: {snippet!r}")
                json_data = resp.json()

                results.append(SearchResult(item=search_item, document=json_data, error=None))
            except Exception as e:
                results.append(SearchResult(item=search_item, document=None, error=str(e)))

        await asyncio.gather(*[asyncio.create_task(fetch_one(it)) for it in items])
        return results

    async def fetch_search_pages_and_docs(
        self,
        *,
        search_params: SearchParams,
        pages: int,
        base_search_url: str = SEARCH_URL,
        base_doc_url: str = DOC_API_URL,
    ) -> list[SearchResult]:
        limits = httpx.Limits(max_connections=self.max_connections, max_keepalive_connections=self.max_connections)
        timeout_cfg = httpx.Timeout(self.timeout)

        async with httpx.AsyncClient(
            headers=HEADERS, limits=limits, timeout=timeout_cfg, follow_redirects=True, http2=True
        ) as client:
            pages_json = await self._search_pages(
                client=client,
                base_search_url=base_search_url,
                search_params=search_params,
                pages=pages,
            )

            all_items = []
            for pj in pages_json:
                all_items.extend(self._extract_items(pj))

            return await self._fetch_docs(client=client, items=all_items, base_doc_url=base_doc_url)
