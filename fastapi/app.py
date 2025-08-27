import os
import sys
import json
import time
import logging
import traceback
from typing import List, Dict, Any, Tuple

import orjson
import redis
import numpy as np
import faiss

from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google import genai
from google.genai import types

from sentence_transformers import SentenceTransformer
from threading import Lock

# ---------------------------
# 기본 설정 / 로거
# ---------------------------
logger = logging.getLogger("easykam")
logging.basicConfig(level=logging.INFO)

app = FastAPI(title="easykam")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # ["https://easykam.life"] 로 바꿀 수 있음
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------
# 환경변수
# ---------------------------
API_KEY = os.environ.get("GOOGLE_API_KEY")
if not API_KEY:
    raise RuntimeError("API KEY가 없습니다. (GOOGLE_API_KEY)")

# RAG 파일 경로 (반드시 docker-compose에서 볼륨/파일 매핑)
DATA_JSONL = os.environ.get("DATA_JSONL", "/app/data/data.jsonl")
FAISS_INDEX_PATH = os.environ.get("FAISS_INDEX_PATH", "/app/data/faiss_index.bin")

# 검색 상수
TOPK = int(os.environ.get("RAG_TOPK", "3"))

# Redis (기존 코드 유지)
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

def _hist_key(session_id: str) -> str:
    return f"chat:{session_id}:messages"

def get_history(session_id: str) -> list[dict]:
    key = f"chat:{session_id}"
    items = r.lrange(key, -HIST_MAX, -1)  # 최신 HIST_MAX개
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
# LLM 클라이언트
# ---------------------------
client = genai.Client(api_key=API_KEY)

# ---------------------------
# 임베딩 / FAISS (글로벌 상태)
# ---------------------------
EMBED_MODEL_NAME = os.environ.get("EMBED_MODEL_NAME", "BAAI/bge-m3")
# bge-m3 권장 인스트럭션(검색/문서 임베딩에 도움)
Q_PREFIX = "Represent this sentence for retrieval: "
D_PREFIX = "Represent this document for retrieval: "

EMBED_MODEL: SentenceTransformer | None = None
FAISS_INDEX: faiss.IndexFlatIP | None = None
DOCS: List[Dict[str, Any]] = []   # [{"id","title","type","content"}...]
RAG_LOCK = Lock()

def _load_embed_model() -> SentenceTransformer:
    logger.info("Loading SentenceTransformer: %s", EMBED_MODEL_NAME)
    model = SentenceTransformer(EMBED_MODEL_NAME)
    logger.info("SentenceTransformer loaded: %s", EMBED_MODEL_NAME)
    return model

def _embed_texts(texts: List[str], is_query=False) -> np.ndarray:
    assert EMBED_MODEL is not None
    if is_query:
        texts = [Q_PREFIX + t for t in texts]
    else:
        texts = [D_PREFIX + t for t in texts]
    vecs = EMBED_MODEL.encode(texts, batch_size=8, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True)
    return vecs.astype("float32")

def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    docs = []
    if not os.path.exists(path) or not os.path.isfile(path):
        logger.warning("JSONL이 없습니다: %s", path)
        return docs
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                # 필수 필드 보정
                if "content" not in obj or not obj["content"]:
                    continue
                obj.setdefault("title", obj.get("id", ""))
                obj.setdefault("type", "regulation")
                docs.append(obj)
            except Exception:
                continue
    logger.info("Docs loaded: %d", len(docs))
    return docs

def _build_faiss(docs: List[Dict[str, Any]], index_path: str) -> faiss.IndexFlatIP:
    if not docs:
        raise RuntimeError("빌드할 문서가 없습니다.")

    # 문서 임베딩 (content 사용)
    contents = [d["content"] for d in docs]
    embs = _embed_texts(contents, is_query=False)  # (N, dim), L2 normed
    dim = embs.shape[1]
    index = faiss.IndexFlatIP(dim)  # inner product
    index.add(embs)

    # 저장
    faiss.write_index(index, index_path)
    logger.info("FAISS index built & saved: %s (docs=%d, dim=%d)", index_path, len(docs), dim)
    return index

def _load_faiss(index_path: str) -> faiss.IndexFlatIP:
    if not os.path.exists(index_path) or not os.path.isfile(index_path):
        raise FileNotFoundError(f"FAISS 인덱스가 없습니다: {index_path}")
    index = faiss.read_index(index_path)
    logger.info("FAISS index loaded: %s", index_path)
    return index

def _init_rag_or_die():
    """
    - 인덱스가 있으면 로드
    - 없고 JSONL이 있으면 빌드 후 로드
    - 둘 다 없으면 서비스 종료
    """
    global EMBED_MODEL, FAISS_INDEX, DOCS
    with RAG_LOCK:
        # 임베딩 모델
        EMBED_MODEL = _load_embed_model()

        # 문서
        DOCS = _read_jsonl(DATA_JSONL)

        # 인덱스
        if os.path.isfile(FAISS_INDEX_PATH):
            try:
                FAISS_INDEX = _load_faiss(FAISS_INDEX_PATH)
                return
            except Exception as e:
                logger.error("FAISS 로드 실패: %s", e)

        # 없으면 빌드 시도
        if DOCS:
            try:
                FAISS_INDEX = _build_faiss(DOCS, FAISS_INDEX_PATH)
                return
            except Exception as e:
                logger.error("FAISS 빌드 실패: %s", e)

        # 여기 오면 RAG 사용할 수 없음 → 서비스 중단
        raise RuntimeError(
            f"RAG 초기화 실패: 인덱스({FAISS_INDEX_PATH})/JSONL({DATA_JSONL}) 확인 필요"
        )

def rag_search(query: str, topk: int = TOPK) -> Tuple[str, List[Dict[str, Any]]]:
    """
    질의 임베딩 → FAISS 상위 topk → 컨텍스트 블록 문자열과 문서 메타 반환
    """
    if FAISS_INDEX is None or EMBED_MODEL is None or not DOCS:
        return "", []

    qv = _embed_texts([query], is_query=True)  # (1, dim)
    D, I = FAISS_INDEX.search(qv, topk)        # I: (1, topk) 인덱스
    idxs = I[0].tolist()
    scores = D[0].tolist()

    found = []
    parts = []
    for rank, (i, score) in enumerate(zip(idxs, scores), start=1):
        if i < 0 or i >= len(DOCS):
            continue
        doc = DOCS[i]
        found.append({"rank": rank, "score": float(score), "doc": doc})
        # 컨텍스트 블록(문서 제목과 본문)
        title = doc.get("title") or doc.get("id") or f"문서#{i}"
        body = doc.get("content", "")
        parts.append(f"[{rank}] 제목: {title}\n{body}")

    context_block = "\n\n----------------\n\n".join(parts).strip()
    return context_block, found

# ---------------------------
# 프롬프트 빌더
# ---------------------------
def build_prompt_with_rag(q: str, history: List[Dict[str, Any]], context_block: str | None) -> str:
    """
    검색 결과가 있으면: 문서 위주로 답변 + 관련 법령·사례 “서술”
    없으면: 질문만 넘김
    """
    sys_inst = (
        "당신은 공공기관 내규 안내 도우미입니다. "
        "아래 '참고 문서'가 있으면 그 문서의 내용을 우선적으로 근거로 하여, "
        "질문에 직접적으로 필요한 항목만 구조적으로 설명하세요. "
        "참고 문서에 포함되지 않은 경우에도, 관련 법령과 실제 사례(공개된 판례/보도자료 수준)를 "
        "적절히 보충해 설명하세요. 단, 법령/사례는 링크나 정확한 명칭을 함께 제시하세요. "
        "불확실하면 명확히 불확실하다고 말하고 추가 질문을 제안하세요."
    )

    parts = [f"[시스템]\n{sys_inst}"]

    if context_block:
        parts.append("\n[참고 문서]\n")
        parts.append(context_block)

    if history:
        parts.append("\n[대화기록(최신 하단)]")
        for m in history:
            who = "사용자" if m.get("role") == "user" else "어시스턴트"
            parts.append(f"- {who}: {m.get('text','').strip()}")

    parts.append("\n[질문]")
    parts.append(q.strip())

    parts.append(
        "\n[지시]\n"
        "- 불필요한 장황한 설명은 피하고 항목별로 정리해서 답하세요.\n"
        "- 가능한 경우 관련 법령과 사례를 함께 제시하세요(링크/정확 명칭 포함).\n"
        "- 참고 문서의 핵심 조항/절차를 그대로 인용하지 말고, 질문 맥락에 맞게 재서술하세요."
    )

    return "\n".join(parts)

# ---------------------------
# API 모델
# ---------------------------
class AskIn(BaseModel):
    question: str
    temperature: float | None = 0.2

class AskOut(BaseModel):
    answer: str

# ---------------------------
# FastAPI 핸들러
# ---------------------------
@app.on_event("startup")
def on_startup():
    try:
        _init_rag_or_die()
    except Exception as e:
        logger.error("RAG 초기화 중 예외: %s", e)
        # 전체 스택 출력
        traceback.print_exc()
        # 서비스 기동 중단
        sys.exit(1)

@app.post("/api/ask", response_model=AskOut)
def ask(payload: AskIn, x_session_id: str = Header(default="")):
    q = (payload.question or "").strip()
    if not q:
        raise HTTPException(400, "질문이 비어 있습니다.")
    if not x_session_id:
        raise HTTPException(400, "세션이 없습니다. X-Session-Id 헤더를 보내주세요.")

    try:
        # 1) 히스토리
        history = get_history(x_session_id)

        # 2) RAG 검색
        context_block, _found = rag_search(q, topk=TOPK)
        has_context = bool(context_block)

        # 3) 프롬프트
        prompt = build_prompt_with_rag(q, history, context_block if has_context else None)

        # 4) LLM 호출
        resp = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            generation_config=types.GenerateContentConfig(
                temperature=payload.temperature or 0.2
            ),
        )
        answer = getattr(resp, "text", "") or ""

        # 5) 히스토리 저장
        append_history(x_session_id, "user", q)
        append_history(x_session_id, "assistant", answer)

        return AskOut(answer=answer)

    except Exception as e:
        logger.error("ASK 처리 중 오류: %s", e)
        traceback.print_exc()
        raise HTTPException(500, f"서버 오류: {e}")

@app.get("/api/check")
def health():
    return {
        "ok_gemini": bool(API_KEY),
        "ok_redis": True
    }

@app.get("/api/hello")
def hello(name: str | None = None):
    return {"msg": f"hi {name}" if name else "hi"}

@app.get("/api/diag")
def diag():
    info = {
        "has_api_key": bool(API_KEY),
        "data_jsonl": DATA_JSONL,
        "faiss_index": FAISS_INDEX_PATH,
        "docs": len(DOCS),
        "index_ready": FAISS_INDEX is not None,
    }
    try:
        info["redis_ping"] = r.ping()
    except Exception as e:
        info["redis_ping"] = f"ERROR: {e}"
    return info
