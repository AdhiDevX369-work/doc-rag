import re
from enum import Enum
from config import BOOK_METADATA, FOLLOWUP_MAX_WORDS, TOPIC_OVERLAP_THRESHOLD


class QueryIntent(Enum):
    SPECIFIC_BOOK = "specific_book"
    CROSS_BOOK = "cross_book"
    COMPARISON = "comparison"
    STRUCTURE = "structure"
    FOLLOWUP = "followup"
    LIST_BOOKS = "list_books"
    META = "meta"
    GENERAL = "general"


def _build_book_patterns() -> dict:
    patterns = {}
    for meta in BOOK_METADATA.values():
        title = meta["title"]
        author = meta["author"]
        
        title_words = [w.lower() for w in title.split() if len(w) > 3]
        key_words = " ".join(title_words[:3])
        patterns[re.compile(rf'\b{re.escape(key_words)}\b', re.I)] = title
        
        if author and author != "Unknown" and author != "Various":
            for a in author.split(","):
                a = a.strip()
                if len(a) > 3:
                    patterns[re.compile(rf'\b{re.escape(a.lower())}\b', re.I)] = title
    
    return patterns


BOOK_PATTERNS = _build_book_patterns()

STRUCTURE_PATTERNS = [re.compile(p, re.I) for p in [
    r'how many chapter', r'\bchapter\b', r'\bchapters\b',
    r'table of contents', r'\btoc\b', r'\boutline\b',
    r'what.*(section|part)', r'list.*(section|part)',
]]

CROSS_BOOK_PATTERNS = [re.compile(p, re.I) for p in [
    r'across.*book', r'each book', r'every book',
    r'both book', r'different book', r'from all', r'in all',
]]

LIST_BOOKS_PATTERNS = [re.compile(p, re.I) for p in [
    r'list.*book', r'all.*book', r'what book', r'which book.*have',
    r'show.*book', r'available book', r'books you have',
    r'how many book', r'more book', r'other book',
]]

META_PATTERNS = [re.compile(p, re.I) for p in [
    r'what.*can you.*help', r'what.*you.*do', r'what.*help.*with',
    r'who are you', r'what are you', r'your capabilities',
    r'how.*can.*you.*assist', r'what.*problems?.*help',
    r'what.*you.*know', r'what.*topics?', r'what.*questions?.*answer',
]]

COMPARISON_PATTERNS = [re.compile(p, re.I) for p in [
    r'\bcompare\b', r'\bdifference\b', r'\bvs\.?\b', r'\bversus\b',
    r'better than', r'which.*better',
]]

FOLLOWUP_INDICATORS = [re.compile(p, re.I) for p in [
    r'^then\b', r'^and\b', r'^also\b', r'^what about\b',
    r'^how about\b', r'^what are\b', r'^what is\b',
    r'^list\b', r'^show\b', r'^tell me\b',
    r'\bthe book\b', r'\bthis book\b', r'\bwhich book\b',
    r'\?$', r'^why\b', r'^how\b', r'^is\b', r'^which\b',
    r'\bmore\b.*\b(detail|explain|info)',
    r'^explain\b', r'^elaborate\b',
    r'^please\b', r'please$',
    r'^can you\b', r'^could you\b',
    r'\bthese\b', r'\bthis\b', r'\bthat\b', r'\bthose\b',
    r'\bsource\b', r'\breference\b', r'\bcite\b',
]]

TECH_TERM_PATTERN = re.compile(r'\b[A-Z]{2,}\b|\b[A-Z][a-z]+(?:[A-Z][a-z]+)+\b')


def extract_book_reference(query: str) -> str:
    for pattern, book_title in BOOK_PATTERNS.items():
        if pattern.search(query):
            return book_title
    return ""


def get_active_book_context(history: list) -> str:
    for entry in reversed(history[-5:]):
        ctx = entry.get("book_context", "")
        if ctx:
            return ctx
    return ""


def has_structure_intent(query: str) -> bool:
    return any(p.search(query) for p in STRUCTURE_PATTERNS)


def has_cross_book_intent(query: str) -> bool:
    return any(p.search(query) for p in CROSS_BOOK_PATTERNS)


def has_comparison_intent(query: str) -> bool:
    return any(p.search(query) for p in COMPARISON_PATTERNS)


def has_list_books_intent(query: str) -> bool:
    return any(p.search(query) for p in LIST_BOOKS_PATTERNS)


def has_meta_intent(query: str) -> bool:
    return any(p.search(query) for p in META_PATTERNS)


def check_topic_relevance(query: str, history: list) -> bool:
    if not history:
        return False
    
    last = history[-1] if history else {}
    prev_answer = last.get("assistant", "")
    
    if not prev_answer:
        return False
    
    query_terms = set(TECH_TERM_PATTERN.findall(query))
    prev_terms = set(TECH_TERM_PATTERN.findall(prev_answer[:400]))
    
    if not prev_terms or not query_terms:
        return False
    
    overlap = len(query_terms & prev_terms) / len(prev_terms)
    return overlap >= TOPIC_OVERLAP_THRESHOLD


def is_followup(query: str, history: list) -> bool:
    if not history:
        return False
    
    word_count = len(query.split())
    
    if any(p.search(query) for p in FOLLOWUP_INDICATORS):
        if word_count <= 10:
            return True
    
    if word_count <= FOLLOWUP_MAX_WORDS:
        return True
    
    return False


def detect_query_intent(query: str, history: list) -> tuple[QueryIntent, str]:
    query_lower = query.lower()
    active_book = get_active_book_context(history)
    
    if has_meta_intent(query):
        return QueryIntent.META, ""
    
    if has_list_books_intent(query):
        return QueryIntent.LIST_BOOKS, ""
    
    explicit_book = extract_book_reference(query)
    if explicit_book:
        if has_structure_intent(query):
            return QueryIntent.STRUCTURE, explicit_book
        return QueryIntent.SPECIFIC_BOOK, explicit_book
    
    if has_cross_book_intent(query):
        return QueryIntent.CROSS_BOOK, ""
    
    if has_comparison_intent(query):
        return QueryIntent.COMPARISON, ""
    
    if active_book:
        if has_structure_intent(query):
            return QueryIntent.STRUCTURE, active_book
        if is_followup(query, history):
            return QueryIntent.FOLLOWUP, active_book
    
    if has_structure_intent(query):
        return QueryIntent.STRUCTURE, ""
    
    if is_followup(query, history):
        return QueryIntent.FOLLOWUP, active_book
    
    return QueryIntent.GENERAL, ""
