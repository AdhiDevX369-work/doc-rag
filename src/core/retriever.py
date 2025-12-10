import re
import hashlib
import concurrent.futures
from collections import defaultdict
from flashrank import RerankRequest

from config import TOP_K
from core.intent import QueryIntent

BOOK_LIST = [
    "AI Engineering: Building Applications with Foundation Models",
    "Build a Large Language Model From Scratch",
    "LLM Engineers Handbook",
    "LL Books Collection",
    "Machine Learning Basics",
]


def format_source(meta: dict) -> str:
    book = meta.get("book_title", "Unknown")
    parts = [f"📖 {book}"]
    
    if meta.get("content_type") == "table_of_contents":
        parts.append("📑 TOC")
    elif "page" in meta:
        parts.append(f"p.{meta['page']}")
    
    if "section_title" in meta:
        parts.append(f"§ {meta['section_title']}")
    elif "chapter_title" in meta:
        parts.append(f"Ch: {meta['chapter_title']}")
    
    author = meta.get("author", "")
    if author and author != "Unknown":
        parts.append(f"by {author}")
    
    return " | ".join(parts)


def content_hash(text: str, meta: dict) -> str:
    # Include metadata in hash to catch duplicates with same content but different sources
    key = f"{text[:300]}|{meta.get('book_title','')}|{meta.get('page','')}|{meta.get('chapter_title','')}"
    normalized = re.sub(r'\s+', ' ', key.lower().strip())
    return hashlib.md5(normalized.encode()).hexdigest()


def search_book(query: str, vectorstore, book: str, k: int = 5) -> list:
    try:
        results = vectorstore.similarity_search_with_score(
            query, k=k, filter={"book_title": book}
        )
        return [(doc, score) for doc, score in results]
    except Exception:
        return []


def search_books_parallel(query: str, vectorstore, books: list, k_per_book: int = 4) -> list:
    all_results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(books)) as executor:
        futures = {
            executor.submit(search_book, query, vectorstore, book, k_per_book): book
            for book in books
        }
        for future in concurrent.futures.as_completed(futures):
            try:
                all_results.extend(future.result())
            except Exception:
                continue
    return all_results


def deduplicate(results: list) -> list:
    seen_hashes = set()
    seen_content = set()
    unique = []
    
    for doc, score in results:
        # Hash with metadata
        h = content_hash(doc.page_content, doc.metadata)
        
        # Also check content similarity (first 200 chars)
        content_key = re.sub(r'\s+', '', doc.page_content[:200].lower())
        
        if h not in seen_hashes and content_key not in seen_content:
            seen_hashes.add(h)
            seen_content.add(content_key)
            unique.append({
                "id": len(unique),
                "text": doc.page_content,
                "meta": doc.metadata,
                "similarity_score": score
            })
    return unique


def expand_vague_query(query: str, history: list) -> str:
    """Expand vague queries using previous context"""
    vague_patterns = ['that', 'this', 'it', 'those', 'these', 'do that', 'do this', 'kind of thing']
    query_lower = query.lower()
    
    if any(p in query_lower for p in vague_patterns) and history:
        last = history[-1] if history else {}
        prev_query = last.get("user", "")
        # Extract key terms from previous query
        if prev_query:
            return f"{query} {prev_query}"
    return query


def retrieve_context(query: str, vectorstore, reranker, intent: QueryIntent,
                     book_filter: str = "", top_k: int = TOP_K, history: list = None) -> tuple[str, list, dict]:
    results = []
    is_structure = intent == QueryIntent.STRUCTURE
    
    # Expand vague queries
    search_query = expand_vague_query(query, history or [])
    
    if intent in [QueryIntent.CROSS_BOOK, QueryIntent.COMPARISON]:
        results = search_books_parallel(search_query, vectorstore, BOOK_LIST, k_per_book=3)
    
    elif intent in [QueryIntent.SPECIFIC_BOOK, QueryIntent.FOLLOWUP, QueryIntent.STRUCTURE] and book_filter:
        if is_structure:
            toc_query = f"{search_query} table of contents chapters sections"
            toc_filter = {"$and": [
                {"content_type": "table_of_contents"},
                {"book_title": book_filter}
            ]}
            try:
                toc_results = vectorstore.similarity_search_with_score(
                    toc_query, k=top_k * 3, filter=toc_filter
                )
                results.extend(toc_results)
            except Exception:
                pass
        
        regular = vectorstore.similarity_search_with_score(
            search_query, k=top_k * 3, filter={"book_title": book_filter}
        )
        results.extend(regular)
    
    elif is_structure:
        try:
            toc_results = vectorstore.similarity_search_with_score(
                query + " table of contents", k=top_k * 2,
                filter={"content_type": "table_of_contents"}
            )
            results.extend(toc_results)
        except Exception:
            pass
        regular = vectorstore.similarity_search_with_score(query, k=top_k * 2)
        results.extend(regular)
    
    else:
        results = vectorstore.similarity_search_with_score(query, k=top_k * 4)
    
    if not results:
        return "", [], {"books_searched": 0, "books": []}
    
    passages = deduplicate(results)
    if not passages:
        return "", [], {"books_searched": 0, "books": []}
    
    reranked = reranker.rerank(RerankRequest(query=query, passages=passages))
    
    if is_structure:
        for doc in reranked:
            if doc["meta"].get("content_type") == "table_of_contents":
                doc["score"] = doc["score"] * 2.0
    
    sorted_docs = sorted(reranked, key=lambda x: x["score"], reverse=True)
    
    if intent in [QueryIntent.CROSS_BOOK, QueryIntent.COMPARISON]:
        book_counts = defaultdict(int)
        top_docs = []
        for doc in sorted_docs:
            book = doc["meta"].get("book_title", "Unknown")
            if book_counts[book] < 2:
                top_docs.append(doc)
                book_counts[book] += 1
            if len(top_docs) >= top_k:
                break
    else:
        top_docs = sorted_docs[:top_k]
    
    sources = []
    context_parts = []
    books_used = set()
    
    for i, doc in enumerate(top_docs, 1):
        book = doc["meta"].get("book_title", "Unknown")
        books_used.add(book)
        sources.append({
            "source": format_source(doc["meta"]),
            "score": doc["score"],
            "metadata": doc["meta"]
        })
        context_parts.append(f"[Source {i} - {book}]\n{doc['text']}")
    
    stats = {"books_searched": len(books_used), "books": list(books_used)}
    return "\n\n---\n\n".join(context_parts), sources, stats
