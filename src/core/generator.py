import torch
from config import SYSTEM_PROMPT, BOOK_METADATA
from core.intent import QueryIntent
from core.validator import validate_response, correction_prompt


def get_book_list_response() -> str:
    lines = ["Here are the books available in my knowledge base:\n"]
    for i, (_, meta) in enumerate(BOOK_METADATA.items(), 1):
        lines.append(f"{i}. **{meta['title']}** by {meta['author']} ({meta['publisher']})")
    lines.append("\nYou can ask me questions about any of these books.")
    return "\n".join(lines)


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


def build_user_prompt(query: str, context: str, stats: dict, history: list = None) -> str:
    if not context:
        return f"Question: {query}\n\nNo relevant context found."
    
    books = ", ".join(stats.get("books", []))
    
    # For vague followups, include previous exchange for context
    prev_context = ""
    if history and len(history) > 0:
        last = history[-1]
        if last.get("assistant"):
            prev_context = f"\nPrevious discussion:\nUser asked: {last.get('user', '')}\nAssistant answered: {last.get('assistant', '')[:500]}...\n"
    
    return f"""Answer ONLY from context below. No external knowledge.

Sources: {books}
{prev_context}
---
{context}
---

Question: {query}

Rules:
1. Answer ONLY from context
2. If question references "that", "it", "this" - refer to previous discussion
3. Cite book and page/chapter
4. For "how to" questions - provide practical steps from the book"""


def _generate(messages: list, model, tokenizer) -> str:
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
    
    return tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def generate_response(query: str, context: str, sources: list, model, tokenizer,
                      history: list, intent: QueryIntent, stats: dict) -> str:
    if intent == QueryIntent.LIST_BOOKS:
        return get_book_list_response()
    
    book_context = stats.get("books", [""])[0] if stats.get("books") else ""
    
    system_prompt = build_system_prompt(intent, book_context)
    user_prompt = build_user_prompt(query, context, stats, history)
    
    messages = [{"role": "system", "content": system_prompt}]
    for turn in history[-3:]:
        messages.append({"role": "user", "content": turn.get("user", "")})
        messages.append({"role": "assistant", "content": turn.get("assistant", "")})
    messages.append({"role": "user", "content": user_prompt})
    
    response = _generate(messages, model, tokenizer)
    
    validation = validate_response(response, context, query)
    
    if not validation.is_valid and validation.confidence < 0.4:
        messages.append({"role": "assistant", "content": response})
        messages.append({"role": "user", "content": correction_prompt(validation.issues, context)})
        response = _generate(messages, model, tokenizer)
    
    return response
