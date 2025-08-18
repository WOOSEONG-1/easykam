FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# ★ CA 인증서 설치 (HTTPS 호출 필수)
RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# 의존성
COPY requirements.txt ./requirements.txt  
RUN pip install --no-cache-dir -r requirements.txt

# 앱 소스
COPY app.py ./app.py                       

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
