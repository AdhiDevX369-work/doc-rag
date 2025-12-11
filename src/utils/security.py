import re
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

MAX_QUERY_LENGTH = 1000
MAX_QUERIES_PER_MINUTE = 10
MAX_QUERIES_PER_HOUR = 100

BLOCKED_PATTERNS = [
    r'<script', r'javascript:', r'on\w+\s*=',
    r'\{\{', r'\}\}', r'<%', r'%>',
    r'__import__', r'eval\s*\(', r'exec\s*\(',
    r'os\.system', r'subprocess', r'import\s+os',
    r'import\s+sys', r'from\s+os', r'open\s*\(',
    r'\bsystem\s*\(', r'popen', r'shell',
]

PROMPT_INJECTION_PATTERNS = [
    r'ignore\s+(all\s+)?(previous|above|prior)\s+(instructions?|prompts?|rules?)',
    r'forget\s+(everything|all|your)\s+(you|instructions?|rules?)',
    r'you\s+are\s+now\s+(a|an|the)',
    r'new\s+instructions?:',
    r'override\s+(system|all|these)\s+(prompt|rules?|instructions?)',
    r'disregard\s+(all|previous|above)\s+(instructions?|prompts?)',
    r'pretend\s+(you\s+are|to\s+be)',
    r'act\s+as\s+(if|a|an)',
    r'roleplay\s+as',
    r'jailbreak',
    r'dan\s+mode',
    r'developer\s+mode',
    r'ignore\s+safety',
    r'bypass\s+(filter|safety|rules?)',
    r'system\s*:\s*you\s+are',
    r'\[system\]',
    r'\[assistant\]',
    r'\[user\]',
]

_compiled_blocked = [re.compile(p, re.IGNORECASE) for p in BLOCKED_PATTERNS]
_compiled_injection = [re.compile(p, re.IGNORECASE) for p in PROMPT_INJECTION_PATTERNS]


@dataclass
class RateLimiter:
    minute_queries: list = field(default_factory=list)
    hour_queries: list = field(default_factory=list)
    
    def check(self) -> tuple[bool, str]:
        now = datetime.now()
        self.minute_queries = [t for t in self.minute_queries if now - t < timedelta(minutes=1)]
        self.hour_queries = [t for t in self.hour_queries if now - t < timedelta(hours=1)]
        
        if len(self.minute_queries) >= MAX_QUERIES_PER_MINUTE:
            return False, "Rate limit: wait a minute."
        if len(self.hour_queries) >= MAX_QUERIES_PER_HOUR:
            return False, "Hourly limit reached."
        
        self.minute_queries.append(now)
        self.hour_queries.append(now)
        return True, ""


def sanitize_input(text: str) -> tuple[str, bool]:
    if not text or not isinstance(text, str):
        return "", False
    
    text = text.strip()
    if len(text) > MAX_QUERY_LENGTH:
        text = text[:MAX_QUERY_LENGTH]
    
    for pattern in _compiled_blocked:
        if pattern.search(text):
            logger.warning(f"Blocked pattern detected in query")
            return "", False
    
    for pattern in _compiled_injection:
        if pattern.search(text):
            logger.warning(f"Prompt injection attempt detected")
            return "", False
    
    text = re.sub(r'[^\w\s\.,\?!;:\'\"\-\(\)\[\]@#\$%\+=/\^]', '', text)
    return text.strip(), bool(text)


def escape_output(text: str) -> str:
    if not text:
        return ""
    return (text
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;"))
