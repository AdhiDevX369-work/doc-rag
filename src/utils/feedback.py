import json
import logging
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)

FEEDBACK_DIR = Path(__file__).parent.parent.parent / "data" / "feedback"
FEEDBACK_DIR.mkdir(parents=True, exist_ok=True)


def log_feedback(query: str, response: str, feedback: str, sources: list = None):
    feedback_file = FEEDBACK_DIR / "feedback.jsonl"
    
    entry = {
        "timestamp": datetime.now().isoformat(),
        "query": query,
        "response": response[:500],
        "feedback": feedback,
        "sources": [s.get("source", "") for s in (sources or [])][:3]
    }
    
    try:
        with open(feedback_file, "a") as f:
            f.write(json.dumps(entry) + "\n")
        logger.info(f"Logged {feedback} feedback for query: {query[:50]}...")
    except Exception as e:
        logger.error(f"Failed to log feedback: {e}")


def get_feedback_stats() -> dict:
    feedback_file = FEEDBACK_DIR / "feedback.jsonl"
    
    if not feedback_file.exists():
        return {"positive": 0, "negative": 0, "total": 0}
    
    positive = 0
    negative = 0
    
    try:
        with open(feedback_file, "r") as f:
            for line in f:
                entry = json.loads(line.strip())
                if entry.get("feedback") == "positive":
                    positive += 1
                elif entry.get("feedback") == "negative":
                    negative += 1
    except Exception:
        pass
    
    return {
        "positive": positive,
        "negative": negative,
        "total": positive + negative,
        "satisfaction": positive / (positive + negative) if (positive + negative) > 0 else 0
    }
