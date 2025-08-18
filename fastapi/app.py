import os
from fastapi import FastAPI, HTTPException
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

@app.post("/api/ask", response_model=AskOut)
def ask(payload: AskIn):
    q = (payload.question or "").strip()
    if not q:
        raise HTTPException(400, "질문이 비어 있습니다.")
    try:
        resp = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=q,
        temperature=payload.temperature,
    )
        return AskOut(answer=resp.text or "")
    except Exception as e:
        # ★ 잠깐만 상세 로그
        import traceback, sys
        traceback.print_exc(file=sys.stderr)
        raise HTTPException(500, f"Gemini 호출 실패: {e}")

@app.get("/api/check")
def health():
    return {"ok": True}

@app.get("/api/hello")
def hello(name: str | None = None):
    return {"msg": f"hi {name}" if name else "hi"}