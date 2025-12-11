import re
import logging
from config import (
    BOOK_METADATA, VALIDATION_THRESHOLD, EVIDENCE_THRESHOLD,
    MAX_VALIDATION_ISSUES, TOPIC_OVERLAP_THRESHOLD, FOLLOWUP_MAX_WORDS,
    DOMAIN_KEYWORDS
)

logger = logging.getLogger(__name__)

SKIP_CLAIM_STARTS = (
    "I don't", "I cannot", "Based on", "I recommend", "You can",
    "For learning", "To learn", "The book", "This book", "I'm",
    "Here are", "You might", "I suggest", "According to"
)

NUMBER_PATTERN = re.compile(r'\b(\d+)\s*(chapter|section|part|step|type|method|approach)s?\b', re.I)
NAME_PATTERN = re.compile(r'\b([A-Z][a-z]+ [A-Z][a-z]+)\b')
WORD_PATTERN = re.compile(r'\b\w{4,}\b')
TECH_TERM_PATTERN = re.compile(r'\b[A-Z]{2,}\b|\b[A-Z][a-z]+(?:[A-Z][a-z]+)+\b')

SPECULATION_PATTERNS = [
    re.compile(r'\blikely\b', re.I),
    re.compile(r'\bprobably\b', re.I),
    re.compile(r'\bmight\b', re.I),
    re.compile(r'\bperhaps\b', re.I),
    re.compile(r'\bpossibly\b', re.I),
    re.compile(r'\bcould be\b', re.I),
    re.compile(r'\bseems to\b', re.I),
    re.compile(r'\bappears to\b', re.I),
    re.compile(r'\bi think\b', re.I),
    re.compile(r'\bi believe\b', re.I),
    re.compile(r'\bmaybe\b', re.I),
    re.compile(r'\bpresumably\b', re.I),
]


class ValidationResult:
    def __init__(self, is_valid: bool, confidence: float, issues: list):
        self.is_valid = is_valid
        self.confidence = confidence
        self.issues = issues


def extract_claims(response: str) -> list[str]:
    sentences = re.split(r'(?<=[.!?])\s+', response)
    claims = []
    
    for s in sentences:
        s = s.strip()
        if len(s) < 15:
            continue
        if s.startswith(SKIP_CLAIM_STARTS):
            continue
        if s.startswith(("- ", "* ", "• ")):
            s = s[2:]
        if re.match(r'^\d+\.\s', s):
            s = re.sub(r'^\d+\.\s', '', s)
        if len(s) > 15:
            claims.append(s)
    
    return claims


def find_evidence(claim: str, context: str, threshold: float = None) -> tuple[bool, float]:
    if threshold is None:
        threshold = EVIDENCE_THRESHOLD
    
    claim_lower = claim.lower()
    context_lower = context.lower()
    
    claim_words = set(WORD_PATTERN.findall(claim_lower))
    context_words = set(WORD_PATTERN.findall(context_lower))
    
    if not claim_words:
        return True, 1.0
    
    overlap = len(claim_words & context_words) / len(claim_words)
    
    if overlap >= threshold:
        claim_tech = set(TECH_TERM_PATTERN.findall(claim))
        context_tech = set(TECH_TERM_PATTERN.findall(context))
        if claim_tech:
            tech_overlap = len(claim_tech & context_tech) / len(claim_tech)
            overlap = (overlap + tech_overlap) / 2
        return overlap >= threshold * 0.8, overlap
    
    best_chunk_score = 0.0
    for chunk in context.split("---"):
        if len(chunk.strip()) < 50:
            continue
        chunk_words = set(WORD_PATTERN.findall(chunk.lower()))
        if claim_words:
            chunk_overlap = len(claim_words & chunk_words) / len(claim_words)
            best_chunk_score = max(best_chunk_score, chunk_overlap)
    
    final_score = max(overlap, best_chunk_score)
    return final_score >= threshold, final_score


def is_query_on_topic(query: str, is_followup: bool = False) -> bool:
    if is_followup:
        return True
    
    query_words = set(query.lower().split())
    domain_overlap = query_words & DOMAIN_KEYWORDS
    
    if domain_overlap:
        return True
    
    tech_terms = set(TECH_TERM_PATTERN.findall(query))
    if tech_terms:
        return True
    
    followup_words = {"these", "this", "that", "those", "it", "them", "book", "source", "reference"}
    if query_words & followup_words:
        return True
    
    question_words = {"what", "how", "why", "when", "where", "which", "who", "explain", "describe"}
    has_question = bool(query_words & question_words)
    
    off_topic_indicators = {
        "weather", "sports", "politics", "celebrity", "movie", "music", "recipe",
        "country", "island", "capital", "population", "geography", "history",
        "president", "king", "queen", "war", "religion", "food", "travel"
    }
    has_off_topic = bool(query_words & off_topic_indicators)
    
    if has_off_topic and not domain_overlap:
        return False
    
    return has_question


def check_context_relevance(query: str, context: str, is_followup: bool = False) -> float:
    if not is_query_on_topic(query, is_followup):
        logger.info(f"Query detected as off-topic: {query[:50]}")
        return 0.0
    
    query_words = set(WORD_PATTERN.findall(query.lower()))
    context_words = set(WORD_PATTERN.findall(context.lower()))
    
    if not query_words:
        return 0.5
    
    word_overlap = len(query_words & context_words) / len(query_words)
    
    query_tech = set(TECH_TERM_PATTERN.findall(query))
    context_tech = set(TECH_TERM_PATTERN.findall(context))
    
    tech_overlap = 0.5
    if query_tech:
        tech_overlap = len(query_tech & context_tech) / len(query_tech)
    
    return (word_overlap + tech_overlap) / 2


NUMBER_WORDS = {
    'one': '1', 'two': '2', 'three': '3', 'four': '4', 'five': '5',
    'six': '6', 'seven': '7', 'eight': '8', 'nine': '9', 'ten': '10',
    'eleven': '11', 'twelve': '12', 'first': '1', 'second': '2', 'third': '3'
}


def normalize_number(text: str) -> str:
    text_lower = text.lower()
    for word, digit in NUMBER_WORDS.items():
        text_lower = re.sub(rf'\b{word}\b', digit, text_lower)
    return text_lower


def check_number_accuracy(response: str, context: str) -> list[str]:
    issues = []
    normalized_context = normalize_number(context)
    response_numbers = NUMBER_PATTERN.findall(response)
    context_numbers = NUMBER_PATTERN.findall(normalized_context)
    
    if response_numbers and not context_numbers:
        issues.append(f"Claimed {response_numbers[0][0]} {response_numbers[0][1]}s without source")
    
    for num, unit in response_numbers[:2]:
        found = any(num == cn[0] and unit.lower() == cn[1].lower() for cn in context_numbers)
        if not found and context_numbers:
            issues.append(f"Number {num} {unit}s not verified in context")
    
    return issues[:2]


def check_names_accuracy(response: str, context: str, query: str = "") -> list[str]:
    issues = []
    response_names = set(NAME_PATTERN.findall(response))
    context_names = set(NAME_PATTERN.findall(context))
    query_names = set(NAME_PATTERN.findall(query)) if query else set()
    
    skip = {
        "The Book", "This Chapter", "For Example", "In This", "Chapter One",
        "Machine Learning", "Deep Learning", "Neural Network", "Natural Language",
        "Large Language", "New York", "San Francisco", "United States"
    }
    
    for name in response_names:
        if name in skip or name in context_names or name in query_names:
            continue
        words = name.split()
        if all(w in context for w in words):
            continue
        issues.append(f"Name '{name}' not found in context")
    
    return issues[:2]


def check_false_attribution(response: str, query: str, context: str) -> list[str]:
    issues = []
    
    nationality_patterns = [
        (r'is (?:a |the )?(\w+) (?:AI engineer|engineer|expert|scientist)', 
         r'(\w+) (?:AI engineer|engineer|expert|scientist)'),
    ]
    
    for resp_pattern, query_pattern in nationality_patterns:
        resp_match = re.search(resp_pattern, response, re.IGNORECASE)
        query_match = re.search(query_pattern, query, re.IGNORECASE)
        
        if resp_match and query_match:
            resp_nat = resp_match.group(1).lower()
            query_nat = query_match.group(1).lower() if query_match else ""
            
            if resp_nat != query_nat and query_nat:
                issues.append(f"Response claims '{resp_nat}' but query asks about '{query_nat}'")
    
    return issues


def check_book_title_accuracy(response: str) -> list[str]:
    issues = []
    
    actual_titles = {meta["title"].lower() for meta in BOOK_METADATA.values()}
    
    book_patterns = [
        r'"([^"]{15,})"(?:\s+\(O\'Reilly\)|\s+published|\s+by\s+\w+)',
        r'(?:book|titled?|called|wrote|written)\s+"([^"]{15,})"',
        r'(?:her|his)\s+book\s+"([^"]{15,})"',
        r'author of\s+"([^"]{15,})"',
    ]
    
    for pattern in book_patterns:
        matches = re.finditer(pattern, response, re.IGNORECASE)
        for match in matches:
            mentioned_title = match.group(1).lower().strip()
            
            title_found = any(
                mentioned_title in actual or actual in mentioned_title
                for actual in actual_titles
            )
            
            if not title_found:
                issues.append(f"Mentioned book '{match.group(1)}' not in source collection")
    
    return issues


def check_topic_continuity(response: str, query: str, history: list) -> list[str]:
    issues = []
    
    if len(query.split()) > FOLLOWUP_MAX_WORDS or not history:
        return issues
    
    last = history[-1] if history else {}
    prev_answer = last.get("assistant", "")
    
    if not prev_answer:
        return issues
    
    prev_terms = set(TECH_TERM_PATTERN.findall(prev_answer[:400]))
    resp_terms = set(TECH_TERM_PATTERN.findall(response[:400]))
    
    if prev_terms and resp_terms:
        overlap = len(prev_terms & resp_terms) / len(prev_terms) if prev_terms else 0
        if overlap < TOPIC_OVERLAP_THRESHOLD:
            logger.warning(f"Topic shift: {resp_terms} vs previous {prev_terms}")
            issues.append("Topic shifted from previous conversation")
    
    return issues


def check_speculation(response: str) -> list[str]:
    issues = []
    for pattern in SPECULATION_PATTERNS:
        match = pattern.search(response)
        if match:
            issues.append(f"Speculative language: '{match.group()}'")
            if len(issues) >= 2:
                break
    return issues


def validate_response(response: str, context: str, query: str, history: list = None) -> ValidationResult:
    if not response:
        return ValidationResult(True, 1.0, [])
    
    if not context.strip():
        return ValidationResult(False, 0.0, ["No source context available"])
    
    if response.startswith("I don't have") or response.startswith("I cannot"):
        return ValidationResult(True, 1.0, [])
    
    query_lower = query.lower()
    skip_patterns = ['suggest', 'recommend', 'should i read', 'best book for']
    if any(p in query_lower for p in skip_patterns):
        return ValidationResult(True, 0.75, [])
    
    issues = []
    claims = extract_claims(response)
    
    if not claims:
        return ValidationResult(True, 1.0, [])
    
    supported = 0
    total_conf = 0.0
    
    for claim in claims:
        found, conf = find_evidence(claim, context)
        if found:
            supported += 1
        else:
            if len(issues) < MAX_VALIDATION_ISSUES:
                issues.append(f"Unsupported: {claim[:60]}...")
        total_conf += conf
    
    issues.extend(check_number_accuracy(response, context))
    issues.extend(check_names_accuracy(response, context, query))
    issues.extend(check_false_attribution(response, query, context))
    issues.extend(check_book_title_accuracy(response))
    issues.extend(check_topic_continuity(response, query, history or []))
    issues.extend(check_speculation(response))
    
    ratio = supported / len(claims) if claims else 1.0
    avg_conf = total_conf / len(claims) if claims else 1.0
    
    final_conf = (ratio * 0.6 + avg_conf * 0.4)
    is_valid = final_conf >= VALIDATION_THRESHOLD and len(issues) <= MAX_VALIDATION_ISSUES
    
    if not is_valid:
        logger.warning(f"Validation failed: conf={final_conf:.2f}, issues={len(issues)}")
    
    return ValidationResult(is_valid, final_conf, issues[:MAX_VALIDATION_ISSUES])


def correction_prompt(issues: list, context: str) -> str:
    issues_text = "\n".join(f"- {i}" for i in issues[:3])
    return f"""Previous answer had issues:
{issues_text}

Rewrite using ONLY the context. Say "I don't have that information" if unsure.

Context:
{context[:2500]}

Corrected answer:"""
