import logging
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from flashrank import Ranker

from config import DB_DIR, MODEL_NAME, EMBEDDING_MODEL, CHROMA_COLLECTION

logger = logging.getLogger(__name__)
_cache = {}

TRUST_REMOTE_CODE = os.getenv("TRUST_REMOTE_CODE", "false").lower() == "true"


def get_model():
    if "model" not in _cache:
        config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
        )
        _cache["tokenizer"] = AutoTokenizer.from_pretrained(MODEL_NAME)
        
        load_kwargs = {
            "quantization_config": config,
            "device_map": "auto",
        }
        
        if TRUST_REMOTE_CODE:
            logger.warning("Loading model with trust_remote_code=True")
            load_kwargs["trust_remote_code"] = True
        
        _cache["model"] = AutoModelForCausalLM.from_pretrained(MODEL_NAME, **load_kwargs)
        logger.info(f"Model loaded: {MODEL_NAME}")
    return _cache["model"], _cache["tokenizer"]


def get_vectorstore():
    if "vectorstore" not in _cache:
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={'device': 'cuda'},
            encode_kwargs={'normalize_embeddings': True},
        )
        _cache["vectorstore"] = Chroma(
            collection_name=CHROMA_COLLECTION,
            persist_directory=str(DB_DIR),
            embedding_function=embeddings,
        )
    return _cache["vectorstore"]


def get_reranker():
    if "reranker" not in _cache:
        _cache["reranker"] = Ranker(
            model_name="ms-marco-MultiBERT-L-12",
            cache_dir="./cache"
        )
    return _cache["reranker"]
