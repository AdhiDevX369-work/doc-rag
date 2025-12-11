import hashlib
import json
import logging
from pathlib import Path
from datetime import datetime, timedelta
from collections import OrderedDict

logger = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).parent.parent.parent / "cache" / "query_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
MAX_CACHE_SIZE = 500
CACHE_TTL_HOURS = 24


class QueryCache:
    def __init__(self, max_size: int = MAX_CACHE_SIZE):
        self.max_size = max_size
        self.cache: OrderedDict = OrderedDict()
        self._load_cache()
    
    def _cache_key(self, query: str, book_filter: str = "") -> str:
        normalized = f"{query.lower().strip()}|{book_filter.lower().strip()}"
        return hashlib.sha256(normalized.encode()).hexdigest()[:16]
    
    def _load_cache(self):
        cache_file = CACHE_DIR / "cache.json"
        if cache_file.exists():
            try:
                with open(cache_file, "r") as f:
                    data = json.load(f)
                    now = datetime.now()
                    for key, entry in data.items():
                        ts = datetime.fromisoformat(entry["timestamp"])
                        if now - ts < timedelta(hours=CACHE_TTL_HOURS):
                            self.cache[key] = entry
                logger.info(f"Loaded {len(self.cache)} cached queries")
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}")
    
    def _save_cache(self):
        cache_file = CACHE_DIR / "cache.json"
        try:
            with open(cache_file, "w") as f:
                json.dump(dict(self.cache), f)
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")
    
    def get(self, query: str, book_filter: str = "") -> dict | None:
        key = self._cache_key(query, book_filter)
        if key in self.cache:
            entry = self.cache[key]
            ts = datetime.fromisoformat(entry["timestamp"])
            if datetime.now() - ts < timedelta(hours=CACHE_TTL_HOURS):
                self.cache.move_to_end(key)
                logger.info(f"Cache hit for query: {query[:50]}...")
                return entry
            else:
                del self.cache[key]
        return None
    
    def set(self, query: str, book_filter: str, response: str, sources: list):
        key = self._cache_key(query, book_filter)
        self.cache[key] = {
            "query": query,
            "book_filter": book_filter,
            "response": response,
            "sources": sources,
            "timestamp": datetime.now().isoformat()
        }
        self.cache.move_to_end(key)
        
        while len(self.cache) > self.max_size:
            self.cache.popitem(last=False)
        
        self._save_cache()
        logger.info(f"Cached response for: {query[:50]}...")
    
    def clear(self):
        self.cache.clear()
        self._save_cache()
        logger.info("Cache cleared")


_cache_instance = None

def get_cache() -> QueryCache:
    global _cache_instance
    if _cache_instance is None:
        _cache_instance = QueryCache()
    return _cache_instance
