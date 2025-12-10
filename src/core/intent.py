import re
from enum import Enum


class QueryIntent(Enum):
    SPECIFIC_BOOK = "specific_book"
    CROSS_BOOK = "cross_book"
    COMPARISON = "comparison"
    STRUCTURE = "structure"
    FOLLOWUP = "followup"
    GENERAL = "general"


BOOK_PATTERNS = {
    r'ai engineer|chip huyen|foundation model': 'AI Engineering: Building Applications with Foundation Models',
    r'build.*llm|large language model.*scratch|sebastian raschka': 'Build a Large Language Model From Scratch',
    r'llm.*handbook|paul iusztin|maxime labonne': 'LLM Engineers Handbook',
    r'machine learning.*basic': 'Machine Learning Basics',
    r'll books': 'LL Books Collection',
}

STRUCTURE_PATTERNS = [
    r'how many chapter', r'chapter', r'chapters',
    r'table of contents', r'toc', r'outline',
    r'what.*(section|part)', r'list.*(section|part)',
]

CROSS_BOOK_PATTERNS = [
    r'all.*book', r'across.*book', r'each book',
    r'every book', r'both book', r'different book',
    r'from all', r'in all',
]

COMPARISON_PATTERNS = [
    r'compare', r'difference', r'vs\.?', r'versus',
    r'better than', r'which.*better',
]

FOLLOWUP_INDICATORS = [
    r'^then\b', r'^and\b', r'^also\b', r'^what about\b',
    r'^how about\b', r'^what are\b', r'^what is\b',
    r'^list\b', r'^show\b', r'^tell me\b',
    r'\bit\b', r'\bits\b', r'\bthis\b', r'\bthat\b',
    r'\bthe book\b', r'\bthis book\b',
]


def extract_book_reference(query: str) -> str:
    query_lower = query.lower()
    for pattern, book_title in BOOK_PATTERNS.items():
        if re.search(pattern, query_lower):
            return book_title
    return ""


def get_active_book_context(history: list) -> str:
    for entry in reversed(history[-5:]):
        ctx = entry.get("book_context", "")
        if ctx:
            return ctx
    return ""


def has_structure_intent(query: str) -> bool:
    return any(re.search(p, query.lower()) for p in STRUCTURE_PATTERNS)


def has_cross_book_intent(query: str) -> bool:
    return any(re.search(p, query.lower()) for p in CROSS_BOOK_PATTERNS)


def has_comparison_intent(query: str) -> bool:
    return any(re.search(p, query.lower()) for p in COMPARISON_PATTERNS)


def is_followup(query: str) -> bool:
    query_lower = query.lower().strip()
    if len(query.split()) <= 8:
        if any(re.search(p, query_lower) for p in FOLLOWUP_INDICATORS):
            return True
    return False


def detect_query_intent(query: str, history: list) -> tuple[QueryIntent, str]:
    query_lower = query.lower()
    active_book = get_active_book_context(history)
    
    explicit_book = extract_book_reference(query)
    if explicit_book:
        if has_structure_intent(query):
            return QueryIntent.STRUCTURE, explicit_book
        return QueryIntent.SPECIFIC_BOOK, explicit_book
    
    if has_cross_book_intent(query):
        return QueryIntent.CROSS_BOOK, ""
    
    if has_comparison_intent(query):
        return QueryIntent.COMPARISON, ""
    
    if active_book and (is_followup(query) or has_structure_intent(query)):
        if has_structure_intent(query):
            return QueryIntent.STRUCTURE, active_book
        return QueryIntent.FOLLOWUP, active_book
    
    if has_structure_intent(query):
        return QueryIntent.STRUCTURE, ""
    
    return QueryIntent.GENERAL, ""
