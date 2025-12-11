import re
import hashlib
import logging
import concurrent.futures
from collections import defaultdict
from rank_bm25 import BM25Okapi
from flashrank import RerankRequest

from config import TOP_K, BOOK_METADATA, FOLLOWUP_MAX_WORDS
from core.intent import QueryIntent

logger = logging.getLogger(__name__)

_bm25_index = {}


def get_book_list() -> list[str]:
    return [meta["title"] for meta in BOOK_METADATA.values()]


def build_bm25_index(vectorstore) -> dict:
    global _bm25_index
    if _bm25_index:
        return _bm25_index
    
    try:
        all_docs = vectorstore._collection.get(include=["documents", "metadatas"])
        documents = all_docs.get("documents", [])
        metadatas = all_docs.get("metadatas", [])
        
        if not documents:
            return {}
        
        tokenized = [doc.lower().split() for doc in documents]
        bm25 = BM25Okapi(tokenized)
        
        _bm25_index = {
            "bm25": bm25,
            "documents": documents,
            "metadatas": metadatas
        }
        logger.info(f"Built BM25 index with {len(documents)} documents")
        return _bm25_index
    except Exception as e:
        logger.warning(f"Failed to build BM25 index: {e}")
        return {}


def bm25_search(query: str, bm25_data: dict, k: int = 10) -> list:
    if not bm25_data:
        return []
    
    bm25 = bm25_data["bm25"]
    documents = bm25_data["documents"]
    metadatas = bm25_data["metadatas"]
    
    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)
    
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    
    results = []
    for idx in top_indices:
        if scores[idx] > 0:
            results.append({
                "text": documents[idx],
                "meta": metadatas[idx],
                "bm25_score": scores[idx]
            })
    return results


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
        for future in concurrent.futures.as_completed(futures, timeout=10.0):
            try:
                all_results.extend(future.result(timeout=5.0))
            except (concurrent.futures.TimeoutError, Exception) as e:
                logger.warning(f"Search timeout or error: {e}")
                continue
    return all_results


def deduplicate(results: list) -> list:
    seen_hashes = set()
    seen_content = set()
    unique = []
    
    for doc, score in results:
        h = content_hash(doc.page_content, doc.metadata)
        
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


def expand_vague_query(query: str, history: list, intent: QueryIntent = None) -> str:
    query_lower = query.lower()
    
    vague_indicators = {"these", "this", "that", "those", "it", "them", "things"}
    has_vague = bool(set(query_lower.split()) & vague_indicators)
    
    reference_patterns = [r'\bwhich book\b', r'\bwhat book\b', r'\bthe book\b', r'\bsource\b']
    has_reference = any(re.search(p, query_lower) for p in reference_patterns)
    
    word_count = len(query.split())
    is_short = word_count <= 10
    
    if not history:
        return query
    
    if not (has_vague or has_reference or is_short):
        return query
    
    last = history[-1] if history else {}
    prev_query = last.get("user", "")
    prev_answer = last.get("assistant", "")[:500]
    
    if not prev_answer:
        return query
    
    key_terms = set()
    
    for text in [prev_query, prev_answer]:
        acronyms = re.findall(r'\b[A-Z]{2,6}\b', text)
        key_terms.update(acronyms[:3])
        
        camel = re.findall(r'\b[A-Z][a-z]+(?:[A-Z][a-z]+)+\b', text)
        key_terms.update(camel[:3])
        
        important = re.findall(r'\b(?:pipeline|system|model|training|inference|feature|architecture|MVP|steps?|building)\b', text, re.I)
        key_terms.update([t.lower() for t in important[:4]])
    
    if key_terms:
        terms_str = ' '.join(list(key_terms)[:5])
        expanded = f"{query} {terms_str}"
        logger.info(f"Expanded query: '{query}' -> '{expanded}'")
        return expanded
    
    if prev_query and is_short:
        expanded = f"{prev_query} {query}"
        logger.info(f"Prepended prev query: '{expanded}'")
        return expanded
    
    return query


def retrieve_context(query: str, vectorstore, reranker, intent: QueryIntent,
                     book_filter: str = "", top_k: int = TOP_K, history: list = None) -> tuple[str, list, dict]:
    results = []
    is_structure = intent == QueryIntent.STRUCTURE
    search_query = expand_vague_query(query, history or [], intent)
    book_list = get_book_list()
    
    if intent in [QueryIntent.CROSS_BOOK, QueryIntent.COMPARISON]:
        results = search_books_parallel(search_query, vectorstore, book_list, k_per_book=3)
    
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
    
    bm25_data = build_bm25_index(vectorstore)
    bm25_results = bm25_search(search_query, bm25_data, k=top_k * 2)
    
    if not results and not bm25_results:
        return "", [], {"books_searched": 0, "books": []}
    
    passages = deduplicate(results)
    
    seen_texts = {p["text"][:200] for p in passages}
    for bm25_doc in bm25_results:
        if bm25_doc["text"][:200] not in seen_texts:
            passages.append({
                "id": len(passages),
                "text": bm25_doc["text"],
                "meta": bm25_doc["meta"],
                "similarity_score": 0.5
            })
            seen_texts.add(bm25_doc["text"][:200])
    
    if not passages:
        return "", [], {"books_searched": 0, "books": []}
    
    reranked = reranker.rerank(RerankRequest(query=search_query, passages=passages))
    reranked = [doc for doc in reranked if doc["score"] >= 0.1]
    
    if reranked:
        scores = [doc["score"] for doc in reranked]
        logger.info(f"Rerank scores - min: {min(scores):.3f}, max: {max(scores):.3f}, avg: {sum(scores)/len(scores):.3f}, count: {len(scores)}")
    
    if not reranked:
        return "", [], {"books_searched": 0, "books": []}
    
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
