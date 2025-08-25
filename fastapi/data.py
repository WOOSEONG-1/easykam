# -*- coding: utf-8 -*-
"""
기존 MD 파일 → JSONL 재생성 파이프라인

- id/title : 파일명에서 개정일 제거 후 원제목
- type     : 파일명 또는 문서 첫 줄에 포함된 키워드 ("규정" / "세칙" / "요강")
- content  : MD 본문 전문 (엔티티 디코딩 처리)

사용 경로:
  - MD_DIR : 기존 변환된 마크다운 폴더
  - JSONL_PATH : 새로 생성할 JSONL 파일
"""

import os
import re
import json
import pathlib
import html
import traceback

# ====== 사용자 경로 ======
MD_DIR      = r"..\markdown"     # 기존 MD들이 들어있는 폴더
JSONL_PATH  = r"..\data_refined.jsonl"
# =========================

ENCODING = "utf-8"
ROOT = pathlib.Path(__file__).parent.resolve()

def to_abs(p: str) -> pathlib.Path:
    P = pathlib.Path(p)
    return P if P.is_absolute() else (ROOT / P)

MD_DIR_P   = to_abs(MD_DIR)
JSONL_P    = to_abs(JSONL_PATH)

JSONL_P.parent.mkdir(parents=True, exist_ok=True)

# ---------- 파일명/타입 정리 ----------
def clean_basename(name: str) -> str:
    """
    파일명(제목)에서 (YYYY-MM-DD ...), (YYYY.M.D ...), (YYYY.M ...개정/제정/시행) 괄호부 제거.
    """
    s = name
    s = re.sub(r"\(\s*\d{4}[-./]\d{1,2}([-.\/]\d{1,2})?[^)]*\)", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def detect_type(title: str, content: str) -> str:
    """
    규정/세칙/요강 중 하나를 type으로 분류.
    - 우선 파일명(title)에서 검색, 없으면 문서 첫 줄에서 검색.
    """
    for kw in ["규정", "세칙", "요강"]:
        if kw in title:
            return kw
    first_line = content.splitlines()[0] if content.strip() else ""
    for kw in ["규정", "세칙", "요강"]:
        if kw in first_line:
            return kw
    return "기타"

# ---------- 메인 ----------
def main():
    mds = sorted(pathlib.Path(MD_DIR_P).glob("*.md"))
    if not mds:
        print(f"[WARN] {MD_DIR_P} 에 MD 파일이 없습니다.")
        return

    # 새 JSONL 덮어쓰기
    if JSONL_P.exists():
        JSONL_P.unlink()

    for md in mds:
        try:
            raw_title = clean_basename(md.stem)
            content = md.read_text(encoding=ENCODING).strip()

            # 엔티티 디코딩 (&#8231; → 실제 문자)
            content = html.unescape(content)

            doc_type = detect_type(raw_title, content)

            rec = {
                "id": raw_title,
                "title": raw_title,
                "type": doc_type,
                "content": content
            }

            with JSONL_P.open("a", encoding=ENCODING) as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

            print(f"[OK] {md.name} → type={doc_type}")

        except Exception as e:
            print(f"[ERROR] {md.name}: {e}")
            traceback.print_exc()

    print(f"\n[DONE] JSONL 생성 완료: {JSONL_P}")

if __name__ == "__main__":
    main()
