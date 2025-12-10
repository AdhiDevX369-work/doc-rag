import re
from difflib import SequenceMatcher


class ValidationResult:
    def __init__(self, is_valid: bool, confidence: float, issues: list):
        self.is_valid = is_valid
        self.confidence = confidence
        self.issues = issues


def extract_claims(response: str) -> list[str]:
    sentences = re.split(r'[.!?]\s+', response)
    claims = []
    skip_starts = ("I don't", "I cannot", "Based on", "I recommend", "You can", 
                   "For learning", "To learn", "The book", "This book")
    for s in sentences:
        s = s.strip()
        if len(s) > 30 and not s.startswith(skip_starts):
            claims.append(s)
    return claims


def find_evidence(claim: str, context: str, threshold: float = 0.35) -> tuple[bool, float]:
    claim_lower = claim.lower()
    context_lower = context.lower()
    
    claim_words = set(re.findall(r'\b\w{4,}\b', claim_lower))
    context_words = set(re.findall(r'\b\w{4,}\b', context_lower))
    
    if not claim_words:
        return True, 1.0
    
    overlap = len(claim_words & context_words) / len(claim_words)
    
    if overlap >= threshold:
        return True, overlap
    
    for chunk in context.split("---"):
        ratio = SequenceMatcher(None, claim_lower[:100], chunk.lower()[:500]).ratio()
        if ratio > 0.3:
            return True, max(overlap, ratio)
    
    return False, overlap


def check_number_accuracy(response: str, context: str) -> list[str]:
    issues = []
    response_numbers = re.findall(r'\b(\d+)\s*(chapter|section|part)s?\b', response.lower())
    context_numbers = re.findall(r'\b(\d+)\s*(chapter|section|part)s?\b', context.lower())
    
    if response_numbers and not context_numbers:
        issues.append(f"Mentioned {response_numbers[0][0]} {response_numbers[0][1]}s without source")
    return issues


def check_names_accuracy(response: str, context: str) -> list[str]:
    issues = []
    response_names = set(re.findall(r'\b([A-Z][a-z]+ [A-Z][a-z]+)\b', response))
    context_names = set(re.findall(r'\b([A-Z][a-z]+ [A-Z][a-z]+)\b', context))
    
    skip = {"The Book", "This Chapter", "For Example", "In This", "Chapter One"}
    for name in response_names:
        if name not in context_names and name not in skip:
            issues.append(f"Name '{name}' not in context")
    return issues


def validate_response(response: str, context: str, query: str) -> ValidationResult:
    if not response or not context:
        return ValidationResult(True, 1.0, [])
    
    # Skip validation for recommendation/suggestion queries
    query_lower = query.lower()
    skip_patterns = ['suggest', 'recommend', 'which book', 'what book', 'should i', 'best book']
    if any(p in query_lower for p in skip_patterns):
        return ValidationResult(True, 0.8, [])
    
    issues = []
    claims = extract_claims(response)
    
    # If no factual claims, response is valid
    if not claims:
        return ValidationResult(True, 1.0, [])
    
    supported = 0
    total_conf = 0.0
    
    for claim in claims:
        found, conf = find_evidence(claim, context)
        if found:
            supported += 1
        else:
            issues.append(f"Unsupported: {claim[:50]}...")
        total_conf += conf
    
    issues.extend(check_number_accuracy(response, context))
    issues.extend(check_names_accuracy(response, context))
    
    ratio = supported / len(claims)
    avg_conf = total_conf / len(claims)
    
    final_conf = (ratio + avg_conf) / 2
    is_valid = final_conf >= 0.4 and len(issues) <= 3
    
    return ValidationResult(is_valid, final_conf, issues)


def correction_prompt(issues: list, context: str) -> str:
    issues_text = "\n".join(f"- {i}" for i in issues[:3])
    return f"""Previous answer had issues:
{issues_text}

Rewrite using ONLY the context. Say "I don't have that information" if unsure.

Context:
{context[:2500]}

Corrected answer:"""
