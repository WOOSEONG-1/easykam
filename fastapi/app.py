from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="easykam")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # 보안 강화하려면 도메인만: ["https://easykam.life"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api/check")
def health():
    return {"ok": True}

@app.get("/api/hello")
def hello(name: str | None = None):
    return {"msg": f"hi {name}" if name else "hi"}