import re
from datetime import datetime, timedelta
from dataclasses import dataclass, field

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
    
    for pattern in BLOCKED_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            return "", False
    
    text = re.sub(r'[^\w\s\.,\?!;:\'\"-]', '', text)
    return text.strip(), bool(text)


def escape_output(text: str) -> str:
    if not text:
        return ""
    return (text
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;"))
