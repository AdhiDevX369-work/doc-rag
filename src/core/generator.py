import logging
import torch
from config import SYSTEM_PROMPT, BOOK_METADATA, MAX_RETRY_ATTEMPTS, CONTEXT_RELEVANCE_THRESHOLD
from core.intent import QueryIntent
from core.validator import validate_response, correction_prompt, check_context_relevance

logger = logging.getLogger(__name__)


def get_book_list_response() -> str:
    lines = ["Here are the books available in my knowledge base:\n"]
    for i, (_, meta) in enumerate(BOOK_METADATA.items(), 1):
        lines.append(f"{i}. **{meta['title']}** by {meta['author']} ({meta['publisher']})")
    lines.append("\nYou can ask me questions about any of these books.")
    return "\n".join(lines)


def get_meta_response() -> str:
    book_list = "\n".join(f"- {meta['title']}" for meta in BOOK_METADATA.values())
    return f"""I'm a document Q&A assistant specialized in answering questions about the following books:

{book_list}

I can help you with:
- Questions about content, concepts, and techniques from these books
- Comparing information across different books
- Finding specific chapters or sections
- Explaining topics covered in these books

Ask me anything about these books and I'll answer based on their content."""


def build_system_prompt(intent: QueryIntent, book_context: str) -> str:
    base = SYSTEM_PROMPT
    
    if intent == QueryIntent.STRUCTURE:
        base += """

CRITICAL: For structure questions:
- ONLY list chapters/sections EXACTLY as shown in TOC
- Do NOT invent or guess chapter names
- If TOC incomplete, say "Based on available TOC..."
- Quote titles exactly"""
    
    elif intent in [QueryIntent.SPECIFIC_BOOK, QueryIntent.FOLLOWUP] and book_context:
        base += f"\n\nFocus ONLY on: {book_context}. Ignore other books."
    
    elif intent == QueryIntent.CROSS_BOOK:
        base += "\n\nSynthesize from ALL books. Cite each book."
    
    return base


def build_user_prompt(query: str, context: str, stats: dict, sources: list, history: list = None) -> str:
    if not context:
        return f"Question: {query}\n\nNo relevant context found. Respond: 'I don't have information about that in these books.'"
    
    source_books = {}
    for src in sources:
        title = src.get("metadata", {}).get("book_title", "")
        if title and title not in source_books:
            for meta in BOOK_METADATA.values():
                if meta["title"] == title:
                    source_books[title] = f"{meta['title']} by {meta['author']} ({meta['publisher']})"
                    break
    
    books = ", ".join(stats.get("books", []))
    source_metadata = "\n".join(f"- {info}" for info in source_books.values()) if source_books else "N/A"
    
    prev_context = ""
    if history and len(history) > 0:
        last = history[-1]
        if last.get("assistant"):
            prev_context = f"\nPrevious discussion:\nUser asked: {last.get('user', '')}\nAssistant answered: {last.get('assistant', '')[:500]}...\n"
    
    return f"""Answer ONLY from context below. No external knowledge allowed.

=== SOURCE BOOKS (These are the ACTUAL source books - use these titles when citing) ===
{source_metadata}

{prev_context}
---
CONTEXT FROM BOOKS:
{context}
---

Question: {query}

STRICT RULES:
1. Answer ONLY from the context above - nothing else
2. When citing books, use the EXACT titles from "SOURCE BOOKS" section above
3. Do NOT mention other books that may be referenced IN the content (like author's previous works)
4. If the context doesn't answer the question, say "I don't have information about that in these books"
5. NEVER confuse book authors with query subjects
6. Cite book title (from SOURCE BOOKS) and page/chapter for all claims"""


def _generate(messages: list, model, tokenizer, stream: bool = False):
    from threading import Thread
    from transformers import TextIteratorStreamer
    
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    
    if stream:
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        generation_kwargs = {
            **inputs,
            "max_new_tokens": 800,
            "temperature": 0.2,
            "do_sample": True,
            "top_p": 0.9,
            "repetition_penalty": 1.1,
            "pad_token_id": tokenizer.eos_token_id,
            "streamer": streamer,
        }
        thread = Thread(target=model.generate, kwargs=generation_kwargs)
        thread.start()
        return streamer
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=800,
            temperature=0.2,
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    return tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def generate_response(query: str, context: str, sources: list, model, tokenizer,
                      history: list, intent: QueryIntent, stats: dict) -> str:
    if intent == QueryIntent.LIST_BOOKS:
        return get_book_list_response()
    
    if intent == QueryIntent.META:
        return get_meta_response()
    
    if not context.strip():
        logger.info("Empty context - returning no info response")
        return "I don't have information about that in these books."
    
    is_followup = intent == QueryIntent.FOLLOWUP
    relevance = check_context_relevance(query, context, is_followup)
    if relevance < CONTEXT_RELEVANCE_THRESHOLD:
        logger.info(f"Low context relevance ({relevance:.2f})")
        return "I don't have information about that in these books."
    
    book_context = stats.get("books", [""])[0] if stats.get("books") else ""
    
    system_prompt = build_system_prompt(intent, book_context)
    user_prompt = build_user_prompt(query, context, stats, sources, history)
    
    messages = [{"role": "system", "content": system_prompt}]
    for turn in history[-2:]:
        messages.append({"role": "user", "content": turn.get("user", "")})
        messages.append({"role": "assistant", "content": turn.get("assistant", "")[:800]})
    messages.append({"role": "user", "content": user_prompt})
    
    response = _generate(messages, model, tokenizer)
    validation = validate_response(response, context, query, history)
    
    for attempt in range(MAX_RETRY_ATTEMPTS):
        if validation.is_valid:
            break
        
        logger.warning(f"Retry {attempt + 1}/{MAX_RETRY_ATTEMPTS}: confidence={validation.confidence:.2f}, issues={validation.issues}")
        
        messages.append({"role": "assistant", "content": response})
        messages.append({"role": "user", "content": correction_prompt(validation.issues, context)})
        response = _generate(messages, model, tokenizer)
        validation = validate_response(response, context, query, history)
    
    if not validation.is_valid and validation.confidence < 0.35:
        logger.warning(f"Final validation failed: confidence={validation.confidence:.2f}")
        return "I don't have reliable information about that in these books."
    
    return response


def generate_response_stream(query: str, context: str, sources: list, model, tokenizer,
                              history: list, intent: QueryIntent, stats: dict):
    if intent == QueryIntent.LIST_BOOKS:
        yield get_book_list_response()
        return
    
    if intent == QueryIntent.META:
        yield get_meta_response()
        return
    
    if not context.strip():
        yield "I don't have information about that in these books."
        return
    
    is_followup = intent == QueryIntent.FOLLOWUP
    relevance = check_context_relevance(query, context, is_followup)
    if relevance < CONTEXT_RELEVANCE_THRESHOLD:
        yield "I don't have information about that in these books."
        return
    
    book_context = stats.get("books", [""])[0] if stats.get("books") else ""
    
    system_prompt = build_system_prompt(intent, book_context)
    user_prompt = build_user_prompt(query, context, stats, sources, history)
    
    messages = [{"role": "system", "content": system_prompt}]
    for turn in history[-2:]:
        messages.append({"role": "user", "content": turn.get("user", "")})
        messages.append({"role": "assistant", "content": turn.get("assistant", "")[:800]})
    messages.append({"role": "user", "content": user_prompt})
    
    streamer = _generate(messages, model, tokenizer, stream=True)
    full_response = ""
    for token in streamer:
        full_response += token
        yield token
    
    validation = validate_response(full_response, context, query, history)
    if not validation.is_valid and validation.confidence < 0.35:
        logger.warning(f"Stream validation failed: confidence={validation.confidence:.2f}")
