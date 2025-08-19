import os
import orjson
import redis
from typing import List, Dict, Any
from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google import genai
from google.genai import types


app = FastAPI(title="easykam")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # ["https://easykam.life"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

API_KEY = os.environ.get("GOOGLE_API_KEY")

if not API_KEY:
    raise RuntimeError("API KEY가 없습니다.")

client = genai.Client(api_key=API_KEY)

class AskIn(BaseModel):
    question: str
    temperature: float | None = 0.2  # 선택: 답변 다양성 제어

class AskOut(BaseModel):
    answer: str

REDIS_PASSWORD = os.getenv("REDIS_PASSWORD")
REDIS_HOST = os.getenv("REDIS_HOST", "easykam-redis")
REDIS_PORT = os.getenv("REDIS_PORT", "6379")

redis_url = f"redis://:{REDIS_PASSWORD}@{REDIS_HOST}:{REDIS_PORT}/0"
r = redis.Redis.from_url(redis_url)

HIST_MAX = int(os.getenv("CHAT_HISTORY_MAX", "20"))          # 세션당 최근 n개
HIST_TTL = int(os.getenv("CHAT_HISTORY_TTL_SEC", "3600"))  # 기본 0.7일

def _hist_key(session_id: str) -> str:
    return f"chat:{session_id}:messages"

def get_history(session_id: str) -> List[Dict[str, Any]]:
    """
    저장 구조: LPUSH로 최신이 앞에 쌓임 → lrange 후 역순으로 시간순 정렬
    """
    k = _hist_key(session_id)
    items = r.lrange(k, 0, HIST_MAX - 1)
    msgs = [orjson.loads(x) for x in reversed(items)]
    return msgs

def append_history(session_id: str, role: str, text: str) -> None:
    k = _hist_key(session_id)
    payload = {"t": int(time.time()), "role": role, "text": text}
    pipe = r.pipeline()
    pipe.lpush(k, orjson.dumps(payload).decode())
    pipe.ltrim(k, 0, HIST_MAX - 1)
    pipe.expire(k, HIST_TTL)
    pipe.execute()

def build_prompt(system_prompt: str, history: List[Dict[str, Any]], user_q: str, context_block: str = "") -> str:
    """
    SDK 버전에 상관없이 가장 호환성 높은 형태:
    대화기록을 사람이 읽는 포맷으로 묶어 하나의 문자열 프롬프트로 전달
    """
    lines = [f"[시스템]\n{system_prompt}"]
    if context_block:
        lines.append(f"\n[검색근거]\n{context_block}")

    if history:
        lines.append("\n[대화기록(최신 하단)]")
        for m in history:
            who = "사용자" if m.get("role") == "user" else "어시스턴트"
            lines.append(f"- {who}: {m.get('text','').strip()}")

    lines.append("\n[현재질문]")
    lines.append(user_q.strip())
    lines.append("\n[지시]\n위의 [검색근거]와 [대화기록]만을 근거로, "
                 "답변 마지막에 참고한 근거를 [1][2]처럼 표기하세요. 불확실하면 추가 질문을 하세요.")
    return "\n".join(lines)

@app.post("/api/ask", response_model=AskOut)
def ask(payload: AskIn, x_session_id: str = Header(default="")):
    q = (payload.question or "").strip()
    if not q:
        raise HTTPException(400, "질문이 비어 있습니다.")
    if not x_session_id:
        # 프론트는 X-Session-Id를 이미 보내고 있음
        raise HTTPException(400, "세션이 없습니다. X-Session-Id 헤더를 보내주세요.")

    try:
        # 1) 세션 대화 불러오기
        history = get_history(x_session_id)

        # 2) (선택) RAG 컨텍스트 자리. 추후 검색결과를 여기에 문자열로 합쳐 넣으면 됨.
        context_block = ""  # "\n".join([format_snippet(s) for s in retrieved_snippets])

        # 3) 프롬프트 구성 (문자열 하나로)
        prompt = build_prompt(SYSTEM_PROMPT, history, q, context_block)

        # 4) LLM 호출 (구버전 SDK 호환: generation_config/temperature 인자 미사용)
        resp = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )
        answer = getattr(resp, "text", "") or ""

        # 5) 세션 저장
        append_history(x_session_id, "user", q)
        append_history(x_session_id, "assistant", answer)

        return AskOut(answer=answer)

    except Exception as e:
        # 원인 파악이 필요하면 아래 두 줄 잠깐 해제해서 로그 확인하세요.
        # import traceback, sys
        # traceback.print_exc(file=sys.stderr)
        raise HTTPException(500, f"Gemini 호출 실패: {e}")

@app.get("/api/check")
def health():
    return {"ok_gemini": API_KEY, "ok_redis": REDIS_PASSWORD}

@app.get("/api/hello")
def hello(name: str | None = None):
    return {"msg": f"hi {name}" if name else "hi"}