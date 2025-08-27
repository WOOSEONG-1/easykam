import os
import sys
import time
import json
import orjson
import traceback
from typing import List, Dict, Any, Tuple

from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import redis
import faiss
from sentence_transformers import SentenceTransformer

from google import genai
from google.genai import types

# ---------------------------
# App & CORS
# ---------------------------
app = FastAPI(title="easykam")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 배포 시 도메인으로 제한 권장
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------
# Env / Paths
# ---------------------------
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_JSONL = os.getenv("DATA_JSONL", os.path.normpath(os.path.join(ROOT_DIR, "../data/data.jsonl")))
FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", os.path.normpath(os.path.join(ROOT_DIR, "../data/faiss_index.bin")))

EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "BAAI/bge-small-ko")  # ← bge-ko
TOPK = int(os.getenv("RAG_TOPK", "3"))

API_KEY = os.environ.get("GOOGLE_API_KEY")
if not API_KEY:
    raise RuntimeError("GOOGLE_API_KEY 가 없습니다.")

client = genai.Client(api_key=API_KEY)

# ---------------------------
# Logging
# ---------------------------
import logging
logger = logging.getLogger("easykam")
logging.basicConfig(level=logging.INFO)

# ---------------------------
# Redis
# ---------------------------
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD")
REDIS_HOST = os.getenv("REDIS_HOST", "easykam-redis")
REDIS_PORT = os.getenv("REDIS_PORT", "6379")
redis_url = f"redis://:{REDIS_PASSWORD}@{REDIS_HOST}:{REDIS_PORT}/0" if REDIS_PASSWORD else f"redis://{REDIS_HOST}:{REDIS_PORT}/0"
r = redis.Redis.from_url(redis_url)

try:
    r.ping()
    logger.info("Redis OK")
except Exception as e:
    logger.error("Redis 연결 실패: %s", e)

HIST_MAX = int(os.getenv("CHAT_HISTORY_MAX", "20"))
HIST_TTL = int(os.getenv("CHAT_HISTORY_TTL_SEC", "3600"))

def get_history(session_id: str) -> List[Dict[str, str]]:
    key = f"chat:{session_id}"
    items = r.lrange(key, -HIST_MAX, -1)
    msgs: List[Dict[str, str]] = []
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
# RAG Globals
# ---------------------------
EMBED_MODEL: SentenceTransformer | None = None
FAISS_INDEX: faiss.Index | None = None
DOCS: List[Dict[str, Any]] = []   # JSONL 라인들의 리스트 (chunks)

def _load_docs(jsonl_path: str) -> List[Dict[str, Any]]:
    if not os.path.isfile(jsonl_path):
        logger.warning("JSONL이 없습니다: %s", jsonl_path)
        return []
    docs: List[Dict[str, Any]] = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                # 최소 필드 보정
                if "content" in obj and isinstance(obj["content"], str):
                    docs.append(obj)
            except Exception:
                continue
    logger.info("Docs loaded: %d", len(docs))
    return docs

def _init_rag_or_die() -> None:
    """임베딩 모델 / 문서 / FAISS 인덱스 로드. 실패 시 프로세스 종료."""
    global EMBED_MODEL, DOCS, FAISS_INDEX

    # 1) 임베딩 모델
    logger.info("Loading SentenceTransformer: %s", EMBED_MODEL_NAME)
    EMBED_MODEL = SentenceTransformer(EMBED_MODEL_NAME)
    logger.info("SentenceTransformer loaded: %s", EMBED_MODEL_NAME)

    # 2) 문서 로드
    DOCS = _load_docs(DATA_JSONL)
    if not DOCS:
        logger.error("RAG 초기화 실패: 문서가 없습니다 (%s)", DATA_JSONL)
        sys.exit(1)

    # 3) FAISS 인덱스 로드
    if not os.path.exists(FAISS_INDEX_PATH) or not os.path.isfile(FAISS_INDEX_PATH):
        logger.error("RAG 초기화 실패: FAISS 인덱스가 없습니다 (%s)", FAISS_INDEX_PATH)
        sys.exit(1)
    try:
        FAISS_INDEX = faiss.read_index(FAISS_INDEX_PATH)
        logger.info("FAISS index loaded: %s", FAISS_INDEX_PATH)
    except Exception as e:
        logger.error("FAISS 로드 실패: %s", e)
        sys.exit(1)

def encode_query(text: str):
    """bge-ko 쿼리 임베딩 (정규화 권장)"""
    v = EMBED_MODEL.encode([text], normalize_embeddings=True)
    return v

def rag_search(query: str, topk: int = TOPK) -> List[Dict[str, Any]]:
    """FAISS로 topk 문서/청크 반환"""
    if not FAISS_INDEX or not DOCS:
        return []
    qv = encode_query(query)
    D, I = FAISS_INDEX.search(qv, topk)
    hits: List[Dict[str, Any]] = []
    for idx in I[0]:
        if 0 <= idx < len(DOCS):
            hits.append(DOCS[idx])
    return hits

def build_prompt(history: List[Dict[str, str]], q: str, contexts: List[Dict[str, Any]]) -> str:
    """
    초기 프롬프트 정책:
    - 컨텍스트(내규)가 있으면: 해당 문서를 근거로 답변 + 관련 법령/사례(요약·설명·링크) 추가
    - 없으면: 문서 없이 질문만 전달(일반 질의응답)
    """
    parts: List[str] = []
    if contexts:
        parts.append(
            "### 규정 컨텍스트(내규 원문 발췌)\n"
            "아래는 질문과 관련성이 높은 내규 전문/청크입니다. 이 내용을 우선 근거로 답변하세요.\n"
        )
        for i, c in enumerate(contexts, start=1):
            title = c.get("title") or c.get("id") or f"doc-{i}"
            ctype = c.get("type", "regulation")
            content = (c.get("content") or "").strip()
            parts.append(f"[{i}] 제목: {title} ({ctype})\n{content}\n")
        parts.append(
            "### 지시\n"
            "1) 위 컨텍스트를 중심으로 질문에 답하세요.\n"
            "2) 답변 끝부분에 관련 법령/시행령/조례 및 사례를 선정·요약·설명하고, 가능한 경우 링크를 제공하세요.\n"
            "3) 컨텍스트에 포함되지 않은 내용은 추측하지 말고, 모호하면 추가 질문을 제안하세요.\n"
        )
    else:
        parts.append(
            "### 지시\n"
            "컨텍스트가 없으므로 일반 질의응답으로 답하세요. 가능하면 관련 법령/시행령/조례 및 사례도 요약·설명하고 링크를 제시하세요.\n"
        )

    if history:
        parts.append("### 대화 히스토리(최신이 하단)")
        for m in history:
            who = "사용자" if m.get("role") == "user" else "어시스턴트"
            text = (m.get("text") or "").strip()
            parts.append(f"- {who}: {text}")

    parts.append("### 사용자 질문")
    parts.append(q.strip())

    return "\n\n".join(parts)

# ---------------------------
# Pydantic
# ---------------------------
class AskIn(BaseModel):
    question: str
    temperature: float | None = 0.2

class AskOut(BaseModel):
    answer: str

# ---------------------------
# Startup: RAG 반드시 성공해야 기동
# ---------------------------
@app.on_event("startup")
def on_startup():
    try:
        _init_rag_or_die()
    except SystemExit:
        raise
    except Exception as e:
        logger.error("RAG 초기화 중 예외: %s", e)
        sys.exit(1)

# ---------------------------
# Routes
# ---------------------------
@app.post("/api/ask", response_model=AskOut)
def ask(payload: AskIn, x_session_id: str = Header(default="")):
    q = (payload.question or "").strip()
    if not q:
        raise HTTPException(400, "질문이 비어 있습니다.")
    if not x_session_id:
        raise HTTPException(400, "세션이 없습니다. X-Session-Id 헤더를 보내주세요.")

    try:
        history = get_history(x_session_id)
        # RAG 검색
        contexts = rag_search(q, topk=TOPK)

        # 프롬프트 생성
        prompt = build_prompt(history, q, contexts)

        # LLM 호출 (단일 프롬프트)
        resp = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )
        answer = getattr(resp, "text", "") or ""

        # 히스토리 저장
        append_history(x_session_id, "user", q)
        append_history(x_session_id, "assistant", answer)

        return AskOut(answer=answer)

    except Exception as e:
        logger.error("ASK 처리 중 오류: %s", e)
        traceback.print_exc(file=sys.stderr)
        raise HTTPException(500, f"서버 오류: {e}")

@app.get("/api/check")
def health():
    info = {
        "has_api_key": bool(API_KEY),
        "data_jsonl": DATA_JSONL,
        "faiss_index": FAISS_INDEX_PATH,
        "docs_loaded": len(DOCS),
        "embed_model": EMBED_MODEL_NAME,
    }
    # Redis ping
    try:
        info["redis_ping"] = r.ping()
    except Exception as e:
        info["redis_ping"] = f"ERROR: {e}"
    return info

@app.get("/api/hello")
def hello(name: str | None = None):
    return {"msg": f"hi {name}" if name else "hi"}
