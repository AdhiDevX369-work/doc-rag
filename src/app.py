import logging
import streamlit as st

from config import BOOK_METADATA
from utils.security import RateLimiter, sanitize_input, escape_output
from utils.models import get_model, get_vectorstore, get_reranker
from utils.feedback import log_feedback, get_feedback_stats
from utils.cache import get_cache
from core.intent import QueryIntent, detect_query_intent
from core.retriever import retrieve_context
from core.generator import generate_response, generate_response_stream

logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

st.set_page_config(page_title="DocRAG", page_icon="📚", layout="wide")


def init_session():
    defaults = {"messages": [], "history": [], "rate_limiter": RateLimiter()}
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def render_sidebar(chunk_count: int):
    with st.sidebar:
        st.header("📚 DocRAG")
        st.metric("Chunks", chunk_count)
        
        stats = get_feedback_stats()
        if stats["total"] > 0:
            st.metric("Satisfaction", f"{stats['satisfaction']:.0%}")
        
        st.subheader("Books")
        for m in BOOK_METADATA.values():
            st.caption(f"• {m['title']}")
        
        col1, col2 = st.columns(2)
        if col1.button("Clear Chat"):
            st.session_state.messages = []
            st.session_state.history = []
            st.rerun()
        if col2.button("Clear Cache"):
            get_cache().clear()
            st.success("Cache cleared")


def render_history():
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander(f"Sources ({len(msg['sources'])})"):
                    for s in msg["sources"]:
                        st.caption(f"{s['source']} ({s['score']:.2f})")


def process_query(query: str, vectorstore, reranker, model, tokenizer, use_stream: bool = True):
    intent, book_ctx = detect_query_intent(query, st.session_state.history)
    
    cache = get_cache()
    cached = cache.get(query, book_ctx)
    if cached:
        st.markdown(cached["response"])
        return cached["response"], cached.get("sources", []), book_ctx
    
    if intent == QueryIntent.LIST_BOOKS:
        from core.generator import get_book_list_response
        response = get_book_list_response()
        st.markdown(response)
        return response, [], ""
    
    if intent == QueryIntent.META:
        from core.generator import get_meta_response
        response = get_meta_response()
        st.markdown(response)
        return response, [], ""
    
    label = {
        QueryIntent.CROSS_BOOK: "Searching all books...",
        QueryIntent.COMPARISON: "Comparing...",
        QueryIntent.STRUCTURE: f"Looking up structure in {book_ctx}..." if book_ctx else "Looking up structure...",
        QueryIntent.SPECIFIC_BOOK: f"Searching {book_ctx}...",
        QueryIntent.FOLLOWUP: f"Continuing in {book_ctx}...",
        QueryIntent.GENERAL: "Searching...",
    }.get(intent, "Processing...")
    
    with st.spinner(label):
        context, sources, stats = retrieve_context(
            query, vectorstore, reranker, intent, book_filter=book_ctx,
            history=st.session_state.history
        )
    
    if use_stream:
        response_placeholder = st.empty()
        full_response = ""
        for token in generate_response_stream(
            query, context, sources, model, tokenizer,
            st.session_state.history, intent, stats
        ):
            full_response += token
            response_placeholder.markdown(full_response + "▌")
        response_placeholder.markdown(full_response)
        response = escape_output(full_response)
    else:
        response = generate_response(
            query, context, sources, model, tokenizer,
            st.session_state.history, intent, stats
        )
        response = escape_output(response)
        st.markdown(response)
    
    if sources:
        with st.expander(f"Sources ({len(sources)}) from {stats['books_searched']} book(s)"):
            for s in sources:
                st.caption(f"{s['source']} ({s['score']:.2f})")
    
    if "⚠️" not in response:
        cache.set(query, book_ctx, response, sources)
    
    if not book_ctx and sources and intent not in [QueryIntent.CROSS_BOOK, QueryIntent.COMPARISON]:
        books = set(s["metadata"].get("book_title", "") for s in sources if s["metadata"].get("book_title"))
        if len(books) == 1:
            book_ctx = list(books)[0]
    
    return response, sources, book_ctx


def main():
    st.title("📚 Document Q&A")
    init_session()
    
    with st.spinner("Loading..."):
        vectorstore = get_vectorstore()
        reranker = get_reranker()
        model, tokenizer = get_model()
    
    render_sidebar(vectorstore._collection.count())
    render_history()
    
    if query := st.chat_input("Ask about your documents..."):
        ok, msg = st.session_state.rate_limiter.check()
        if not ok:
            st.error(msg)
            return
        
        query, valid = sanitize_input(query)
        if not valid:
            st.error("Invalid input")
            return
        
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)
        
        with st.chat_message("assistant"):
            try:
                response, sources, book_ctx = process_query(
                    query, vectorstore, reranker, model, tokenizer
                )
            except Exception as e:
                logger.error(f"Error processing query: {e}")
                response = "Sorry, I encountered an error processing your question. Please try again."
                sources = []
                book_ctx = ""
                st.error(response)
        
        st.session_state.messages.append({
            "role": "assistant", "content": response, "sources": sources
        })
        if "⚠️" not in response and "error" not in response.lower():
            st.session_state.history.append({
                "user": query, "assistant": response, "book_context": book_ctx
            })
        
        col1, col2, col3 = st.columns([1, 1, 8])
        if col1.button("👍", key=f"pos_{len(st.session_state.messages)}"):
            log_feedback(query, response, "positive", sources)
            st.toast("Thanks for the feedback!")
        if col2.button("👎", key=f"neg_{len(st.session_state.messages)}"):
            log_feedback(query, response, "negative", sources)
            st.toast("Thanks for the feedback!")


if __name__ == "__main__":
    main()
