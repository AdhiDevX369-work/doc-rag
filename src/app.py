import streamlit as st

from config import BOOK_METADATA
from utils.security import RateLimiter, sanitize_input, escape_output
from utils.models import get_model, get_vectorstore, get_reranker
from core.intent import QueryIntent, detect_query_intent
from core.retriever import retrieve_context
from core.generator import generate_response

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
        
        st.subheader("Books")
        for m in BOOK_METADATA.values():
            st.caption(f"• {m['title']}")
        
        if st.button("Clear Chat"):
            st.session_state.messages = []
            st.session_state.history = []
            st.rerun()


def render_history():
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander(f"Sources ({len(msg['sources'])})"):
                    for s in msg["sources"]:
                        st.caption(f"{s['source']} ({s['score']:.2f})")


def process_query(query: str, vectorstore, reranker, model, tokenizer):
    intent, book_ctx = detect_query_intent(query, st.session_state.history)
    
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
            query, vectorstore, reranker, intent, book_filter=book_ctx
        )
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
            response, sources, book_ctx = process_query(
                query, vectorstore, reranker, model, tokenizer
            )
        
        st.session_state.messages.append({
            "role": "assistant", "content": response, "sources": sources
        })
        st.session_state.history.append({
            "user": query, "assistant": response, "book_context": book_ctx
        })


if __name__ == "__main__":
    main()
