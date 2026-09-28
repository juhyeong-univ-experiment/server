"""External book-data sources used by the real-time enrichment pipeline.

- Open Library (no key): bibliographic data + work description + subjects
- Wikipedia ko/en (no key): search + page summary (acts as the web-search source)
- Google Books (optional, GOOGLE_BOOKS_API_KEY): description/categories
"""
from __future__ import annotations

import asyncio
import logging
import os
import re
from typing import Any
from urllib.parse import quote

import httpx

logger = logging.getLogger(__name__)

HTTP_TIMEOUT = httpx.Timeout(8.0, connect=4.0)
HEADERS = {"User-Agent": "univ-book-reco/1.0 (research prototype)"}


def _has_hangul(text: str) -> bool:
    return bool(re.search(r"[가-힣]", text or ""))


async def _get_json(client: httpx.AsyncClient, url: str, params: dict[str, Any] | None = None) -> Any:
    try:
        resp = await client.get(url, params=params)
        if resp.status_code != 200:
            logger.info("source_http_status url=%s status=%s", url, resp.status_code)
            return None
        return resp.json()
    except Exception as exc:  # network errors must never break the chat pipeline
        logger.info("source_http_error url=%s error=%s", url, exc)
        return None


async def fetch_open_library(client: httpx.AsyncClient, title: str, author: str | None) -> dict[str, Any] | None:
    params = {
        "title": title,
        "limit": 1,
        "fields": "key,title,author_name,first_publish_year,subject,number_of_pages_median,isbn",
    }
    if author:
        params["author"] = author
    data = await _get_json(client, "https://openlibrary.org/search.json", params)
    docs = (data or {}).get("docs") or []
    if not docs:
        return None
    doc = docs[0]
    description = ""
    work = await _get_json(client, f"https://openlibrary.org{doc['key']}.json") if doc.get("key") else None
    if work:
        raw_desc = work.get("description")
        description = raw_desc.get("value", "") if isinstance(raw_desc, dict) else (raw_desc or "")
    return {
        "source": "openlibrary",
        "title": doc.get("title"),
        "author": ", ".join(doc.get("author_name") or []) or None,
        "year": doc.get("first_publish_year"),
        "pages": doc.get("number_of_pages_median"),
        "subjects": (doc.get("subject") or [])[:15],
        "description": description,
        "url": f"https://openlibrary.org{doc['key']}" if doc.get("key") else None,
    }


async def fetch_wikipedia(client: httpx.AsyncClient, title: str, author: str | None) -> dict[str, Any] | None:
    langs = ["ko", "en"] if _has_hangul(title) else ["en", "ko"]
    for lang in langs:
        # 1) direct page hit on the exact title (skip disambiguation pages)
        direct = await _get_json(client, f"https://{lang}.wikipedia.org/api/rest_v1/page/summary/{quote(title)}")
        if direct and direct.get("type") == "standard" and len(direct.get("extract") or "") >= 80:
            return {
                "source": f"wikipedia-{lang}",
                "title": direct.get("title"),
                "description": direct["extract"],
                "url": (direct.get("content_urls") or {}).get("desktop", {}).get("page"),
            }
        # 2) full-text search as a lightweight web search
        query = f"{title} {author or ''} 책" if lang == "ko" else f"{title} {author or ''} book"
        search = await _get_json(
            client,
            f"https://{lang}.wikipedia.org/w/api.php",
            {"action": "query", "list": "search", "srsearch": query.strip(), "srlimit": 3, "format": "json"},
        )
        hits = ((search or {}).get("query") or {}).get("search") or []
        for hit in hits:
            page_title = hit.get("title", "")
            summary = await _get_json(
                client, f"https://{lang}.wikipedia.org/api/rest_v1/page/summary/{quote(page_title)}"
            )
            extract = (summary or {}).get("extract") or ""
            if len(extract) < 80:
                continue
            return {
                "source": f"wikipedia-{lang}",
                "title": page_title,
                "description": extract,
                "url": ((summary or {}).get("content_urls") or {}).get("desktop", {}).get("page"),
            }
    return None


async def fetch_google_books(client: httpx.AsyncClient, title: str, author: str | None) -> dict[str, Any] | None:
    api_key = os.getenv("GOOGLE_BOOKS_API_KEY")
    if not api_key:
        return None
    q = f"intitle:{title}" + (f"+inauthor:{author}" if author else "")
    data = await _get_json(
        client, "https://www.googleapis.com/books/v1/volumes", {"q": q, "maxResults": 1, "key": api_key}
    )
    items = (data or {}).get("items") or []
    if not items:
        return None
    info = items[0].get("volumeInfo") or {}
    year = (info.get("publishedDate") or "")[:4]
    return {
        "source": "googlebooks",
        "title": info.get("title"),
        "author": ", ".join(info.get("authors") or []) or None,
        "year": int(year) if year.isdigit() else None,
        "pages": info.get("pageCount"),
        "subjects": info.get("categories") or [],
        "description": info.get("description") or "",
        "url": info.get("infoLink"),
    }


async def collect_book_evidence(
    title: str, author: str | None = None, alt_titles: list[str] | None = None
) -> list[dict[str, Any]]:
    """Query all sources concurrently for each title variant; return those that produced something."""
    titles = [t for t in dict.fromkeys([title, *(alt_titles or [])]) if t]
    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT, headers=HEADERS, follow_redirects=True) as client:
        tasks = []
        for t in titles:
            tasks += [
                fetch_google_books(client, t, author),
                fetch_open_library(client, t, author),
                fetch_wikipedia(client, t, author),
            ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
    evidence = [r for r in results if isinstance(r, dict) and r]
    # de-duplicate identical descriptions coming from several title variants
    seen: set[str] = set()
    unique = []
    for ev in evidence:
        key = (ev.get("source"), (ev.get("description") or "")[:120])
        if key in seen:
            continue
        seen.add(key)
        unique.append(ev)
    return unique
