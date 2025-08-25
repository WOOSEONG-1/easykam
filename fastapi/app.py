# -*- coding: utf-8 -*-
"""
RAG 기반 파이프라인 with bge-ko + FAISS + Gemini
- JSONL에서 regulation 문서를 로드
- bge-ko 임베딩 후 FAISS 인덱스 구축
- 질문 시 검색 → context 포함해 LLM 호출
"""

import os
import faiss
import orjson
import redis
import time
import numpy as np
import traceback, sys, logging

from typing import List, Dict, Any
from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from sentence_transformers import SentenceTransformer
from google import genai


# ---------------------------
# 설정
# ---------------------------
JSONL_PATH = "../data_refined.jsonl"
INDEX_PATH = "faiss_index.bin"
MODEL_NAME = "BAAI/bge-m3"   # 한국어 포함 다국어 bge-m3 (bge-ko 모델 대체 가능)

app = FastAPI(title="easykam")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

API_KEY = os.environ.get("GOOGLE_API_KEY")
if not API_KEY:
    raise RuntimeError("API KEY가 없습니다.")
client = genai.Client(api_key=API_KEY)

logger = logging.getLogger("easykam")
logging.basicConfig(level=logging.INFO)

# ---------------------------
# Redis 연결
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

HIST_MAX = int(os.getenv("CHAT_HISTORY_MAX", "20"))     
HIST_TTL = int(os.getenv("CHAT_HISTORY_TTL_SEC", "3600"))  


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
# 임베딩 + FAISS
# ---------------------------
embedder = SentenceTransformer(MODEL_NAME)

def build_faiss():
    docs = []
    with open(JSONL_PATH, "r", encoding="utf-8") as f:
        for line in f:
            obj = orjson.loads(line)
            docs.append(obj)

    texts = [d["content"] for d in docs]
    vecs = embedder.encode(texts, normalize_embeddings=True)
    dim = vecs.shape[1]

    index = faiss.IndexFlatIP(dim)  # 내적 기반 (cosine 유사도)
    index.add(np.array(vecs, dtype="float32"))
    faiss.write_index(index, INDEX_PATH)
    return docs, index

def load_faiss():
    index = faiss.read_index(INDEX_PATH)
    docs = []
    with open(JSONL_PATH, "r", encoding="utf-8") as f:
        for line in f:
            docs.append(orjson.loads(line))
    return docs, index


def rag_search(query: str, topk: int = 3):
    docs, index = load_faiss()
    q_vec = embedder.encode([query], normalize_embeddings=True)
    D, I = index.search(np.array(q_vec, dtype="float32"), topk)

    hits = [docs[i] for i in I[0] if i < len(docs)]
    if not hits:
        return ""
    context_block = "\n\n---\n\n".join(
        f"[{h['title']}]\n{h['content'][:1200]}..." for h in hits
    )
    return context_block


# ---------------------------
# 프롬프트 생성
# ---------------------------
def build_rag_prompt(history: list[dict], q: str, context_block: str) -> str:
    parts = []
    if context_block.strip():
        parts.append("### 관련 문서:")
        parts.append(context_block.strip())
        parts.append("\n[지시]\n위의 '관련 문서'를 기준으로 답변을 작성하세요. "
                     "문서에 없는 부분은 추측하지 말고, "
                     "관련 법령과 실제 사례를 보완하여 설명하세요. "
                     "출처 문서명을 [제목]으로 표기하세요.")
    else:
        parts.append("### 관련 문서가 없음")
        parts.append("\n[지시]\n관련 문서가 없으므로, 질문에 대한 일반적인 지식만을 바탕으로 답변하세요.")

    if history:
        parts.append("### 대화 히스토리:")
        for m in history:
            who = "사용자" if m.get("role") == "user" else "어시스턴트"
            parts.append(f"{who}: {m['text']}")

    parts.append("### 사용자 질문:")
    parts.append(q.strip())

    return "\n\n".join(parts)


# ---------------------------
# FastAPI 엔드포인트
# ---------------------------
class AskIn(BaseModel):
    question: str
    temperature: float | None = 0.2  

class AskOut(BaseModel):
    answer: str


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


@app.get("/api/check")
def health():
    return {"ok_gemini": API_KEY, "ok_redis": REDIS_PASSWORD}
