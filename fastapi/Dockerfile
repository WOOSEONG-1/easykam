# 가볍고 보안 업데이트 빠른 slim 이미지
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# 런타임에 필요한 OS 패키지(옵션)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 의존성 먼저 설치(캐시 효율)
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# 앱 복사
COPY fastapi/app.py ./app.py

# 컨테이너 내부 포트
EXPOSE 8000

# uvicorn 구동
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
