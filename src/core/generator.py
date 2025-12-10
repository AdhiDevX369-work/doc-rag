import torch
from config import SYSTEM_PROMPT
from core.intent import QueryIntent


def build_system_prompt(intent: QueryIntent, book_context: str) -> str:
    base = SYSTEM_PROMPT
    
    if intent == QueryIntent.STRUCTURE:
        base += """

CRITICAL INSTRUCTIONS FOR STRUCTURE QUESTIONS:
- ONLY list chapters/sections that appear EXACTLY in the provided TOC content
- Count the chapters by looking at the actual numbered items in the TOC
- Do NOT invent, guess, or add any chapters not in the sources
- If TOC is incomplete, say "Based on the available TOC excerpt..."
- Quote chapter titles exactly as written"""
    
    elif intent in [QueryIntent.SPECIFIC_BOOK, QueryIntent.FOLLOWUP]:
        if book_context:
            base += f"\n\nFocus ONLY on: {book_context}. Ignore information from other books."
    
    elif intent == QueryIntent.CROSS_BOOK:
        base += "\n\nSynthesize information from ALL books. Cite each book when referencing."
    
    return base


def build_user_prompt(query: str, context: str, sources: list, stats: dict) -> str:
    if not context:
        return f"Question: {query}\n\nNo relevant context found."
    
    books_info = f"Sources from {stats.get('books_searched', 0)} book(s): {', '.join(stats.get('books', []))}"
    
    return f"""Answer based ONLY on the context below. Do not use external knowledge.

{books_info}

---
{context}
---

Question: {query}

Rules:
1. Answer ONLY from the context above
2. If information is not in context, say "I don't have that information"
3. Cite book title and page/chapter for each fact
4. For chapter questions: count and list ONLY what's in the TOC, do not invent"""


def generate_response(query: str, context: str, sources: list, model, tokenizer,
                      history: list, intent: QueryIntent, stats: dict) -> str:
    book_context = stats.get("books", [""])[0] if stats.get("books") else ""
    
    system_prompt = build_system_prompt(intent, book_context)
    user_prompt = build_user_prompt(query, context, sources, stats)
    
    messages = [{"role": "system", "content": system_prompt}]
    
    for turn in history[-3:]:
        messages.append({"role": "user", "content": turn.get("user", "")})
        messages.append({"role": "assistant", "content": turn.get("assistant", "")})
    
    messages.append({"role": "user", "content": user_prompt})
    
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    
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
    
    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    ).strip()
    
    return response
