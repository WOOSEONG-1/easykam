<template>
  <div class="page">
    <!-- 헤더 -->
    <header class="topbar">
      <div class="frame topbar-inner"><!-- ★ 헤더 내부도 프레임 가운데 정렬 -->
        <h1>easyKam</h1>
      </div>
    </header>

    <!-- 본문 프레임(가운데 정렬 + 반응형) -->
    <main class="frame shell"><!-- ★ frame가 가로 가운데 + 최대폭 캡 -->
      <!-- 채팅 박스 -->
      <section ref="listEl" class="chat">
        <div v-for="(m, i) in messages" :key="i" :class="['row', m.role]">
          <div class="bubble">{{ m.text }}</div>
        </div>

        <div v-if="loading" class="row assistant">
          <div class="bubble dots">
            <span class="d"></span><span class="d"></span><span class="d"></span>
          </div>
        </div>
      </section>

      <!-- 입력창 -->
      <form class="inputbar" @submit.prevent="onSubmit">
        <span class="hash">#</span>
        <input
          v-model.trim="input"
          placeholder="궁금하신 사항을 입력해 주세요."
          @keydown.enter.exact.prevent="onSubmit"
        />
        <button type="submit" :disabled="loading || !input">
          보내기
          <svg viewBox="0 0 24 24" aria-hidden="true">
            <path d="M2 21l21-9L2 3l5 9-5 9zm7-9l-3.5-6.3L19 12 5.5 18.3 9 12z" fill="currentColor"/>
          </svg>
        </button>
      </form>
    </main>

    <footer class="foot">
      <div class="frame foot-inner"><!-- ★ 푸터 내부도 프레임 가운데 정렬 -->
        © {{ year }} easyKam
      </div>
    </footer>
  </div>
</template>

<script setup>
import { onMounted, onUpdated, reactive, ref } from 'vue'

const API = '/api/ask' // 필요하면 절대경로로 교체
const input   = ref('')
const loading = ref(false)
const listEl  = ref(null)
const year    = new Date().getFullYear()

// 세션 (웹/앱 공통)
const sessionId = ref(localStorage.getItem('easykam_session_id') || '')
if (!sessionId.value) {
  sessionId.value = (crypto?.randomUUID?.() || `${Date.now()}-${Math.random()}`)
  localStorage.setItem('easykam_session_id', sessionId.value)
}

// 초기 메시지
const messages = reactive([
  { role: 'assistant', text: '안녕하세요! 궁금한 내용을 입력해 주세요.' }
])

// 스크롤 아래 고정
const scrollDown = () => {
  requestAnimationFrame(() => {
    listEl.value?.scrollTo({ top: listEl.value.scrollHeight, behavior: 'smooth' })
  })
}
onMounted(scrollDown)
onUpdated(scrollDown)

async function send(q) {
  const question = (q ?? input.value).trim()
  if (!question || loading.value) return
  messages.push({ role: 'user', text: question })
  input.value = ''
  loading.value = true
  try {
    const r = await fetch(API, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'X-Session-Id': sessionId.value
      },
      body: JSON.stringify({ question, session_id: sessionId.value })
    })
    if (!r.ok) throw new Error(`HTTP ${r.status}`)
    const data   = await r.json()
    const answer = data?.answer ?? data?.text ?? JSON.stringify(data)
    messages.push({ role: 'assistant', text: String(answer) })
  } catch (e) {
    messages.push({ role: 'assistant', text: `오류가 발생했어요. 잠시 후 다시 시도해 주세요.\n(상세: ${e?.message || e})` })
  } finally {
    loading.value = false
  }
}
function onSubmit(){ send() }
</script>

<!-- 전역 리셋/변수 (scoped 아님) -->
<style>
*,*::before,*::after { box-sizing: border-box; }
html, body, #app { height: 100%; margin: 0; }
body { overflow-x: hidden; -webkit-font-smoothing: antialiased; }

:root{
  /* 안전영역 대응 (노치/홈바) */
  --safe-top: env(safe-area-inset-top, 0px);
  --safe-right: env(safe-area-inset-right, 0px);
  --safe-bottom: env(safe-area-inset-bottom, 0px);
  --safe-left: env(safe-area-inset-left, 0px);

  /* 레이아웃 스페이싱 */
  --gap: clamp(10px, 2.5vw, 18px);
  --pad: clamp(10px, 2.5vw, 18px);

  /* 컴포넌트 높이(반응형) */
  --h-topbar: clamp(52px, 8dvh, 68px);
  --h-input : clamp(56px, 9dvh, 72px);
  --h-foot  : clamp(36px, 6dvh, 48px);

  /* 프레임 최대폭: 모바일 100%, 데스크탑 최대 1200~1280px */
  --frame-max: clamp(360px, 94vw, 1200px);
}

/* ★ 초대형 화면에서도 보기 좋게 상한 다시 캡 */
@media (min-width:1600px){
  .frame{ max-width: 1280px; }     /* 초대형 화면 상한 */
}
</style>

<!-- 컴포넌트 스타일 -->
<style scoped>
/* ★ 화면을 3행 그리드(헤더/본문/푸터)로, 가로는 항상 가운데 */
.page{
  min-height: 100dvh;
  display: grid;
  grid-template-rows: var(--h-topbar) 1fr var(--h-foot);
  background: #f7f8fb; color:#0f172a;
  padding-left: var(--safe-left);
  padding-right: var(--safe-right);
  max-width: 1200px;
  margin-right: auto;
  margin-left: auto;
}

.frame {
  display: block;
  width: 100%;
  max-width: 900px;              /* ★ 최대 폭을 960px 정도로 제한 (원하시면 1024px, 1120px 등으로 조정) */
  margin-left: auto;
  margin-right: auto;
  padding-left: 24px;            /* ★ 좌우 패딩 */
  padding-right: 24px;
}

/* 헤더 */
.topbar{
  height: var(--h-topbar);
  padding-top: var(--safe-top);
  background:#0f172a; color:#fff;
  display:flex; align-items:center;
}
/* ★ 헤더 내부를 확실하게 가운데 정렬 */
.topbar-inner{
  display:flex; align-items:center; justify-content:center;
}
.topbar h1{
  margin: 0;
  font-weight: 700;
  font-size: clamp(16px, 2.6vw, 20px);
  letter-spacing: .3px;
  text-align: center;
}

/* 본문 프레임 */
.shell{
  display: flex;
  flex-direction: column;
  gap: var(--gap);
  /* ★ 남는 세로 공간을 차지해 내부 카드(채팅 박스)가 세로로도 안정 */
  min-height: 0; /* overflow 컨테이너에서 필수 */
  padding-top: var(--gap);
  padding-bottom: var(--gap);
}

/* 채팅 박스: 가로 100%, 세로 자동 계산 */
.chat{
  width: 100%;
  background:#fff;
  border-radius: 16px;
  box-shadow: 0 6px 18px rgba(2,8,23,.06);
  padding: var(--pad);
  overflow-y: auto;
  overflow-x: hidden;
  position: relative;

  /* 화면 높이 - 헤더 - 입력창 - 푸터 - 간격들 - 안전영역 */
  height: calc(
    100dvh - var(--h-topbar) - var(--h-input) - var(--h-foot)
    - (var(--gap) * 2) - (var(--safe-top) + var(--safe-bottom))
  );
  min-height: 36vh;
}

/* 중앙 로고 (kamco.gif, 85% 투명) */
.chat::before{
  content:"";
  position:absolute; top:50%; left:50%;
  transform:translate(-50%,-50%);
  width:min(300px, 55%);
  height:min(300px, 55%);
  background:url("/src/assets/kamco.gif") center/contain no-repeat;
  opacity:.35;
  pointer-events:none;
  z-index:0;
}

/* 말풍선은 로고 위 */
.row{ position:relative; z-index:1; display:flex; margin:.35rem 0; }
.row.user{ justify-content:flex-end; }
.row.assistant{ justify-content:flex-start; }

.bubble{
  max-width: 85%;
  padding:.6rem .8rem;
  border-radius: 1rem;
  white-space: pre-wrap; line-height:1.5;
  word-break: break-word;
}
.user .bubble{ background:#2563eb; color:#fff; }
.assistant .bubble{ background:#f1f5f9; color:#0f172a; }

.bubble.dots{ display:inline-flex; gap:.3rem; }
.d{ width:6px; height:6px; border-radius:999px; background:#94a3b8; animation:bounce 1s infinite; }
.d:nth-child(1){animation-delay:-.2s}.d:nth-child(3){animation-delay:.2s}
@keyframes bounce{0%,80%,100%{transform:translateY(0)}40%{transform:translateY(-4px)}}

/* 입력창: 프레임 안에서 100% */
.inputbar{
  width: 100%;
  height: var(--h-input);
  background:#fff;
  border-radius: 16px;
  box-shadow: 0 6px 18px rgba(2,8,23,.08);
  display: grid;
  grid-template-columns: auto 1fr auto; /* # | input | button */
  align-items: center;
  gap: .5rem;
  padding: .5rem;
  position: sticky;
  bottom: calc(var(--safe-bottom) + 8px);
}
.hash{ padding:.25rem .5rem; color:#64748b; }
.inputbar input{
  min-width: 0;                    /* grid 넘침 방지 */
  width: 100%;
  border: 0; outline: 0;
  background: transparent; color:#0f172a;
  padding: .4rem;
  font-size: clamp(14px, 2.6vw, 16px);
}
.inputbar button{
  white-space: nowrap;
  display:inline-flex; align-items:center; gap:.4rem;
  border:0; background:#2563eb; color:#fff;
  font-weight:700; border-radius:12px;
  padding:.55rem .9rem; cursor:pointer;
  font-size: clamp(14px, 2.6vw, 16px);
}
.inputbar button:disabled{ opacity:.5; cursor:not-allowed; }
.inputbar svg{ width:18px; height:18px; }

/* 푸터 */
.foot{
  height: var(--h-foot);
  display:flex; align-items:center;
  padding-top: .6rem;
  padding-bottom: calc(.6rem + var(--safe-bottom));
  color:#94a3b8; font-size:12px;
}
/* ★ 푸터 내용도 프레임 중앙 정렬 */
.foot-inner{
  display:flex; align-items:center; justify-content:center;
}
</style>
