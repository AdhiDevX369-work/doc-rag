import time
import json
import logging
from pathlib import Path
from datetime import datetime
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict

logger = logging.getLogger(__name__)

METRICS_DIR = Path(__file__).parent.parent.parent / "data" / "metrics"
METRICS_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class QueryMetrics:
    query: str = ""
    intent: str = ""
    retrieval_time_ms: float = 0
    generation_time_ms: float = 0
    total_time_ms: float = 0
    chunks_retrieved: int = 0
    rerank_score_min: float = 0
    rerank_score_max: float = 0
    rerank_score_avg: float = 0
    validation_passed: bool = True
    validation_confidence: float = 1.0
    cache_hit: bool = False
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class MetricsCollector:
    def __init__(self):
        self._current = QueryMetrics()
        self._start_time = None
    
    def start_query(self, query: str, intent: str):
        self._current = QueryMetrics(query=query[:100], intent=intent)
        self._start_time = time.perf_counter()
    
    def set_retrieval_time(self, ms: float):
        self._current.retrieval_time_ms = ms
    
    def set_rerank_scores(self, scores: list):
        if scores:
            self._current.rerank_score_min = min(scores)
            self._current.rerank_score_max = max(scores)
            self._current.rerank_score_avg = sum(scores) / len(scores)
            self._current.chunks_retrieved = len(scores)
    
    def set_validation(self, passed: bool, confidence: float):
        self._current.validation_passed = passed
        self._current.validation_confidence = confidence
    
    def set_cache_hit(self, hit: bool):
        self._current.cache_hit = hit
    
    def end_query(self):
        if self._start_time:
            self._current.total_time_ms = (time.perf_counter() - self._start_time) * 1000
            self._current.generation_time_ms = self._current.total_time_ms - self._current.retrieval_time_ms
        self._log_metrics()
    
    def _log_metrics(self):
        metrics_file = METRICS_DIR / f"metrics_{datetime.now().strftime('%Y%m%d')}.jsonl"
        try:
            with open(metrics_file, "a") as f:
                f.write(json.dumps(asdict(self._current)) + "\n")
        except Exception as e:
            logger.warning(f"Failed to log metrics: {e}")


@contextmanager
def track_time(name: str):
    start = time.perf_counter()
    yield
    elapsed = (time.perf_counter() - start) * 1000
    logger.info(f"{name}: {elapsed:.1f}ms")


def get_daily_stats(date: str = None) -> dict:
    if date is None:
        date = datetime.now().strftime('%Y%m%d')
    
    metrics_file = METRICS_DIR / f"metrics_{date}.jsonl"
    if not metrics_file.exists():
        return {}
    
    total_queries = 0
    cache_hits = 0
    total_time = 0
    validation_passed = 0
    
    try:
        with open(metrics_file, "r") as f:
            for line in f:
                m = json.loads(line.strip())
                total_queries += 1
                if m.get("cache_hit"):
                    cache_hits += 1
                total_time += m.get("total_time_ms", 0)
                if m.get("validation_passed"):
                    validation_passed += 1
    except Exception:
        pass
    
    return {
        "total_queries": total_queries,
        "cache_hit_rate": cache_hits / total_queries if total_queries else 0,
        "avg_response_time_ms": total_time / total_queries if total_queries else 0,
        "validation_pass_rate": validation_passed / total_queries if total_queries else 0
    }


_collector = None

def get_metrics_collector() -> MetricsCollector:
    global _collector
    if _collector is None:
        _collector = MetricsCollector()
    return _collector
