# -*- coding: utf-8 -*-
import os
import sys
import time
import faiss
import orjson
import redis
import numpy as np
import traceback
import logging
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from google import genai

# ---------------------------
# 설정 (경로/모델/ENV)
# ---------------------------
DATA_JSONL_PATH = os.getenv("DATA_JSONL_PATH", "/app/data.jsonl")
FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "/app/faiss_index.bin")
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "BAAI/bge-m3")  # 한국어 포함 다국어 BGE

HIST_MAX = int(os.getenv("CHAT_HISTORY_MAX", "20"))
HIST_TTL = int(os.getenv("CHAT_HISTORY_TTL_SEC", "3600"))

API_KEY = os.environ.get("GOOGLE_API_KEY")
if not API_KEY:
    raise RuntimeError("API KEY가 없습니다.")
client = genai.Client(api_key=API_KEY)

# ---------------------------
# FastAPI / CORS
# ---------------------------
app = FastAPI(title="easykam")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # 필요 시 도메인 제한
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------
# 로거
# ---------------------------
logger = logging.getLogger("easykam")
logging.basicConfig(level=logging.INFO)

# ---------------------------
# Redis
# ---------------------------
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD")
REDIS_HOST = os.getenv("REDIS_HOST", "easykam-redis")
REDIS_PORT = os.getenv("REDIS_PORT", "6379")

redis_url = f"redis://:{REDIS_PASSWORD}@{REDIS_HOST}:{REDIS_PORT}/0"
r = redis.Redis.from_url(redis_url)
try:
    r.ping()
    logger.info("Redis OK")
except Exception as e:
    logger.error("Redis 연결 실패: %s", e)

def get_history(session_id: str) -> list[dict]:
    key = f"chat:{session_id}"
    items = r.lrange(key, -HIST_MAX, -1)
    msgs = []
    for x in items:
        try:
            obj = orjson.loads(x)
            msgs.append({"role": obj.get("role"), "text": obj.get("text") or obj.get("content")})
        except Exception:
            msgs.append({"role": "user", "text": str(x)})
    return msgs

def append_history(session_id: str, role: str, text: str) -> None:
    key = f"chat:{session_id}"
    payload = {"t": int(time.time()), "role": role, "text": text}
    r.rpush(key, orjson.dumps(payload).decode())
    r.ltrim(key, -HIST_MAX, -1)
    r.expire(key, HIST_TTL)

# ---------------------------
# 전역 RAG 상태 (메모리 상주)
# ---------------------------
embedder: Optional[SentenceTransformer] = None
docs_mem: Optional[List[Dict[str, Any]]] = None
faiss_index: Optional[faiss.Index] = None
dim: Optional[int] = None

def _load_docs(path: str) -> List[Dict[str, Any]]:
    docs = []
    if not os.path.exists(path):
        logger.warning("JSONL이 없습니다: %s", path)
        return docs
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                docs.append(orjson.loads(line))
    return docs

def _encode_texts(texts: List[str]) -> np.ndarray:
    vecs = embedder.encode(texts, normalize_embeddings=True)
    return np.asarray(vecs, dtype="float32")

def _build_index(docs: List[Dict[str, Any]]) -> faiss.Index:
    global dim
    if not docs:
        # 비어있는 인덱스도 만들 수 있게 최소 1차원이라도 필요하지만,
        # 여기서는 문서가 없을 때 검색을 스킵하도록 처리
        raise RuntimeError("인덱스 빌드 실패: 문서가 없습니다.")
    texts = [d.get("content", "") for d in docs]
    vecs = _encode_texts(texts)
    dim = vecs.shape[1]
    index = faiss.IndexFlatIP(dim)  # cosine 유사도용
    index.add(vecs)
    faiss.write_index(index, FAISS_INDEX_PATH)
    logger.info("FAISS index built & saved: %s (docs=%d, dim=%d)", FAISS_INDEX_PATH, len(docs), dim)
    return index

def _load_index_or_build():
    """서버 기동 시: 인덱스 파일이 있으면 로드, 없으면 JSONL로부터 빌드."""
    global embedder, docs_mem, faiss_index, dim
    # 임베더 준비
    embedder = SentenceTransformer(EMBED_MODEL_NAME)
    logger.info("SentenceTransformer loaded: %s", EMBED_MODEL_NAME)

    # 문서 로드
    docs_mem = _load_docs(DATA_JSONL_PATH)
    logger.info("Docs loaded: %d", len(docs_mem))

    # 인덱스 로드/빌드
    if os.path.exists(FAISS_INDEX_PATH):
        faiss_index = faiss.read_index(FAISS_INDEX_PATH)
        # 차원 복구용: 하나 질의해서 dim을 채우기보단 모델로부터 얻음
        sample_vec = _encode_texts(["ping"])
        dim = sample_vec.shape[1]
        logger.info("FAISS index loaded: %s (dim=%d)", FAISS_INDEX_PATH, dim)
    else:
        faiss_index = _build_index(docs_mem)

def rag_search(query: str, topk: int = 3) -> str:
    """메모리 상주 인덱스/문서를 사용하여 컨텍스트 문자열 생성."""
    if faiss_index is None or docs_mem is None or embedder is None or not docs_mem:
        return ""
    q_vec = _encode_texts([query])
    D, I = faiss_index.search(q_vec, topk)
    ids = [i for i in I[0] if 0 <= i < len(docs_mem)]
    if not ids:
        return ""
    blocks = []
    for i in ids:
        h = docs_mem[i]
        title = h.get("title") or h.get("id") or "문서"
        text = (h.get("content") or "")[:1200]
        blocks.append(f"[{title}]\n{text}...")
    return "\n\n---\n\n".join(blocks)

def build_rag_prompt(history: list[dict], q: str, context_block: str) -> str:
    parts = []
    if context_block.strip():
        parts.append("### 관련 문서:")
        parts.append(context_block.strip())
        parts.append(
            "\n[지시]\n위의 '관련 문서'를 **핵심 근거**로 답변하세요."
            " 문서의 내용을 바탕으로 구체적으로 설명하고,"
            " 한국 법령과 실제 사례를 **추가로** 보완하세요."
            " 문서에 없는 내용은 단정하지 말고, 필요한 경우 추정이유를 밝히세요."
            " 답변 말미에 사용한 문서명을 대괄호로 한 번 더 요약 표기하세요."
        )
    else:
        parts.append("### 관련 문서가 없음")
        parts.append(
            "\n[지시]\n관련 문서가 없으므로 일반적 배경지식을 바탕으로 답변하세요."
            " 가능한 경우 관련 법령·사례를 함께 제시하세요."
        )

    if history:
        parts.append("### 대화 히스토리:")
        for m in history:
            who = "사용자" if m.get("role") == "user" else "어시스턴트"
            parts.append(f"{who}: {m['text']}")

    parts.append("### 사용자 질문:")
    parts.append(q.strip())
    return "\n\n".join(parts)

# ---------------------------
# 스키마
# ---------------------------
class AskIn(BaseModel):
    question: str
    temperature: float | None = 0.2

class AskOut(BaseModel):
    answer: str

# ---------------------------
# 엔드포인트
# ---------------------------
@app.on_event("startup")
def _on_startup():
    try:
        _load_index_or_build()
    except Exception as e:
        logger.error("RAG 초기화 실패: %s", e)
        # 인덱스 없이도 서비스는 살아있되, RAG만 빈 컨텍스트가 되도록

@app.post("/api/ask", response_model=AskOut)
def ask(payload: AskIn, x_session_id: str = Header(default="")):
    q = (payload.question or "").strip()
    if not q:
        raise HTTPException(400, "질문이 비어 있습니다.")
    if not x_session_id:
        raise HTTPException(400, "세션이 없습니다. X-Session-Id 헤더를 보내주세요.")

    try:
        history = get_history(x_session_id)
        context_block = rag_search(q, topk=3)

        prompt = build_rag_prompt(history, q, context_block)
        resp = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )
        answer = getattr(resp, "text", "") or ""

        append_history(x_session_id, "user", q)
        append_history(x_session_id, "assistant", answer)
        return AskOut(answer=answer)
    except Exception as e:
        logger.error("ASK 처리 중 오류: %s", e)
        traceback.print_exc(file=sys.stderr)
        raise HTTPException(500, f"서버 오류: {e}")

@app.post("/api/reindex")
def reindex():
    """JSONL이 갱신됐을 때 수동 재색인."""
    global docs_mem, faiss_index
    try:
        docs = _load_docs(DATA_JSONL_PATH)
        if not docs:
            raise HTTPException(400, "JSONL 문서가 없습니다.")
        index = _build_index(docs)
        docs_mem = docs
        faiss_index = index
        return {"ok": True, "docs": len(docs)}
    except Exception as e:
        logger.error("재색인 실패: %s", e)
        raise HTTPException(500, f"재색인 실패: {e}")

@app.get("/api/check")
def health():
    return {
        "ok_gemini": bool(API_KEY),
        "ok_redis": True
    }

@app.get("/api/diag")
def diag():
    info = {
        "has_api_key": bool(os.environ.get("GOOGLE_API_KEY")),
        "jsonl_exists": os.path.exists(DATA_JSONL_PATH),
        "faiss_exists": os.path.exists(FAISS_INDEX_PATH),
        "docs_loaded": len(docs_mem or []),
        "index_loaded": faiss_index is not None,
        "embed_model": EMBED_MODEL_NAME,
        "paths": {"jsonl": DATA_JSONL_PATH, "index": FAISS_INDEX_PATH},
    }
    try:
        info["redis_ping"] = r.ping()
    except Exception as e:
        info["redis_ping"] = f"ERROR: {e}"
    return info
