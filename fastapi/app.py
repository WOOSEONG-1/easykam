# app.py
import os
import sys
import json
import time
import logging
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from threading import Thread, Event, Lock

import orjson
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ---- Logging ----
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s:%(name)s:%(message)s",
)
logger = logging.getLogger("easykam")

# ---- Env & Config ----
API_KEY = os.environ.get("GOOGLE_API_KEY", "")
DATA_JSONL = os.environ.get("DATA_JSONL", "/app/data/data.jsonl")
FAISS_INDEX_PATH = os.environ.get("FAISS_INDEX_PATH", "/app/data/faiss.index")
EMBED_MODEL_NAME = os.environ.get("EMBED_MODEL_NAME", "BAAI/bge-m3")

REDIS_HOST = os.environ.get("REDIS_HOST", "")
REDIS_PORT = int(os.environ.get("REDIS_PORT", "6379"))
REDIS_PASSWORD = os.environ.get("REDIS_PASSWORD", "")

TOP_K_DEFAULT = int(os.environ.get("TOP_K", "4"))
TEMPERATURE_DEFAULT = float(os.environ.get("TEMPERATURE", "0.2"))

# ---- Optional deps early import guards ----
try:
    import redis  # type: ignore
except Exception:  # pragma: no cover
    redis = None

try:
    import faiss  # type: ignore
except Exception as e:
    logger.error("FAISS import failed: %s", e)
    raise

try:
    from sentence_transformers import SentenceTransformer  # type: ignore
except Exception as e:
    logger.error("sentence-transformers import failed: %s", e)
    raise

try:
    from google import genai
    from google.genai import types as genai_types
except Exception as e:
    logger.error("google.genai import failed: %s", e)
    raise

# ---- Globals (protected by locks/events) ----
index_ready = Event()
_init_lock = Lock()

_encoder: Optional[SentenceTransformer] = None
_index: Optional[Any] = None  # FAISS index
_docs: List[Dict[str, Any]] = []
_redis = None
_client: Optional[genai.Client] = None

# ---- FastAPI ----
app = FastAPI(title="EasyKAM RAG API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 필요시 도메인으로 제한
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- Data & RAG helpers ----
def _load_docs(path: str) -> List[Dict[str, Any]]:
    """
    Expect each line is a JSON object with at least:
      { "id": "...", "text": "...", "title": "...", ... }
    """
    docs: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = orjson.loads(line)
                if "text" in obj and obj["text"]:
                    docs.append(obj)
            except Exception as e:
                logger.warning("Skip line parse error: %s", e)
    return docs


def _ensure_encoder(name: str) -> SentenceTransformer:
    global _encoder
    if _encoder is None:
        logger.info("Loading SentenceTransformer: %s", name)
        _encoder = SentenceTransformer(name)
        logger.info("SentenceTransformer loaded: %s", name)
    return _encoder


def _build_or_load_faiss(embeds, dim: int, path: str):
    """
    If index file exists, read it. Else build a new index and save.
    """
    if os.path.exists(path):
        try:
            index = faiss.read_index(path)
            logger.info("FAISS index loaded: %s", path)
            return index
        except Exception as e:
            logger.warning("Failed to load FAISS index, will rebuild: %s", e)

    quantizer = faiss.IndexFlatIP(dim)
    index = faiss.IndexIDMap2(quantizer)
    index.add_with_ids(embeds, (faiss.numpy.array(range(embeds.shape[0]))).astype("int64"))
    faiss.write_index(index, path)
    logger.info("FAISS index built and saved: %s", path)
    return index


def _encode_texts(encoder: SentenceTransformer, texts: List[str]):
    import numpy as np  # local import to shorten cold start
    # bge-m3: normalize_embeddings=True recommended for IP similarity
    vecs = encoder.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return np.asarray(vecs, dtype="float32")


def _search_faiss(encoder: SentenceTransformer, index, query: str, top_k: int):
    import numpy as np
    q = _encode_texts(encoder, [query])
    scores, idxs = index.search(q, top_k)
    idxs = idxs[0].tolist()
    scores = scores[0].tolist()
    hits = []
    for i, s in zip(idxs, scores):
        if i < 0 or i >= len(_docs):
            continue
        doc = _docs[i]
        hits.append(
            {
                "rank": len(hits) + 1,
                "score": float(s),
                "id": doc.get("id", i),
                "title": doc.get("title", ""),
                "text": doc.get("text", ""),
            }
        )
    return hits


def _redis_connect():
    global _redis
    if redis is None:
        return None
    try:
        _redis = redis.Redis(
            host=REDIS_HOST or "localhost",
            port=REDIS_PORT,
            password=REDIS_PASSWORD or None,
            decode_responses=True,
            socket_timeout=2,
            socket_connect_timeout=2,
        )
        _redis.ping()
        logger.info("Redis OK")
        return _redis
    except Exception as e:
        logger.warning("Redis not available: %s", e)
        _redis = None
        return None


def _hist_key(session_id: str) -> str:
    return f"easykam:hist:{session_id}"


def _append_history(session_id: str, role: str, content: str):
    if not _redis:
        return
    try:
        key = _hist_key(session_id)
        item = {"role": role, "content": content, "ts": time.time()}
        _redis.rpush(key, orjson.dumps(item).decode("utf-8"))
        # optional trim
        _redis.ltrim(key, -40, -1)
    except Exception as e:
        logger.debug("history append failed: %s", e)


def _get_history(session_id: str, limit: int = 10) -> List[Dict[str, Any]]:
    if not _redis:
        return []
    try:
        key = _hist_key(session_id)
        arr = _redis.lrange(key, -limit, -1) or []
        return [orjson.loads(x) for x in arr]
    except Exception:
        return []

# ---- GenAI client ----
def _ensure_genai_client() -> genai.Client:
    global _client
    if _client is None:
        if not API_KEY:
            raise RuntimeError("GOOGLE_API_KEY is not set")
        _client = genai.Client(api_key=API_KEY)
    return _client

# ---- Background init ----
def _init_rag_or_die():
    """
    Load docs, encoder, embeddings and faiss index.
    Raises on failure.
    """
    global _docs, _index, _encoder

    if not os.path.exists(DATA_JSONL):
        raise FileNotFoundError(f"DATA_JSONL not found: {DATA_JSONL}")

    _docs = _load_docs(DATA_JSONL)
    logger.info("Docs loaded: %d", len(_docs))
    if not _docs:
        raise RuntimeError("No docs loaded")

    enc = _ensure_encoder(EMBED_MODEL_NAME)
    texts = [d.get("text", "") for d in _docs]
    embeds = _encode_texts(enc, texts)
    _index = _build_or_load_faiss(embeds, embeds.shape[1], FAISS_INDEX_PATH)


def _bg_init():
    # Run heavy init in background, do not crash the process on failure
    with _init_lock:
        try:
            _redis_connect()
            _init_rag_or_die()
            index_ready.set()
            logger.info("RAG index ready")
        except Exception as e:
            logger.exception("Background RAG init failed: %s", e)
            # remains not ready; endpoints will return 503


@app.on_event("startup")
def on_startup():
    Thread(target=_bg_init, daemon=True).start()
    logger.info("Startup: kicked off background RAG init")

# ---- Schemas ----
class AskPayload(BaseModel):
    session_id: str = Field(default="default")
    question: str
    top_k: int = Field(default=TOP_K_DEFAULT, ge=1, le=20)
    temperature: float = Field(default=TEMPERATURE_DEFAULT, ge=0.0, le=1.0)

class AskResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]

# ---- Routes ----
@app.get("/api/check")
def check():
    return {
        "ok": True,
        "index_ready": index_ready.is_set(),
        "docs": len(_docs),
        "embed_model": EMBED_MODEL_NAME,
        "has_api_key": API_KEY,
        "REDIS_PASSWORD" : REDIS_PASSWORD
    }

@app.get("/api/diag")
def diag():
    return {
        "index_ready": index_ready.is_set(),
        "docs": len(_docs),
        "embed_model": EMBED_MODEL_NAME,
        "data_jsonl": DATA_JSONL,
        "faiss_index_path": FAISS_INDEX_PATH,
        "redis": bool(_redis),
    }

def _build_prompt(question: str, hits: List[Dict[str, Any]]) -> str:
    context_blocks = []
    for h in hits:
        title = h.get("title", "")
        text = h.get("text", "")
        context_blocks.append(f"[{h['rank']}] {title}\n{text}".strip())
    context = "\n\n".join(context_blocks) if context_blocks else "N/A"

    sys_inst = (
        "당신은 한국자산관리공사(캠코)의 내규에 대해 답하는 비서입니다. "
        "오로지 제공된 자료(context)와 일반 상식 범위 안에서만 대답하세요. "
        "문서 근거가 없거나 확실하지 않으면 '자료상 확인되지 않습니다'라고 말하고, "
        "가능하면 어떤 조항/섹션을 인용해 주세요."
    )
    prompt = (
        f"{sys_inst}\n\n"
        f"# 질문\n{question}\n\n"
        f"# 자료(Context)\n{context}\n\n"
        f"# 지침\n"
        f"- 한국어로 간결하게 답변\n"
        f"- 문서에서 인용한 부분은 따옴표로 표시\n"
        f"- 답변 마지막에 근거 문서 랭크 번호를 괄호로 요약 예: (근거: 1,3)\n"
    )
    return prompt

@app.post("/api/ask", response_model=AskResponse)
def ask(payload: AskPayload):
    if not index_ready.is_set():
        raise HTTPException(status_code=503, detail="RAG 초기화 중입니다. 잠시 후 다시 시도해주세요.")

    if not payload.question or not payload.question.strip():
        raise HTTPException(status_code=400, detail="question is required")

    try:
        encoder = _ensure_encoder(EMBED_MODEL_NAME)
        hits = _search_faiss(encoder, _index, payload.question, payload.top_k)
    except Exception as e:
        logger.exception("Search error: %s", e)
        raise HTTPException(status_code=500, detail="search failed")

    # Compose prompt
    prompt = _build_prompt(payload.question, hits)

    # LLM call
    try:
        client = _ensure_genai_client()
        resp = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=genai_types.GenerateContentConfig(
                temperature=payload.temperature
            ),
        )
        # google.genai returns .text in newer SDKs
        answer_text = getattr(resp, "text", None) or (resp.candidates[0].content.parts[0].text if getattr(resp, "candidates", None) else "")
    except Exception as e:
        logger.exception("ASK 처리 중 오류: %s", e)
        raise HTTPException(status_code=502, detail="LLM call failed")

    # Save history (best-effort)
    try:
        _append_history(payload.session_id, "user", payload.question)
        _append_history(payload.session_id, "assistant", answer_text)
    except Exception:
        pass

    return AskResponse(answer=answer_text, sources=hits)

# ---- Root (optional)
@app.get("/")
def root():
    return {"service": "easykam-api", "ok": True}
