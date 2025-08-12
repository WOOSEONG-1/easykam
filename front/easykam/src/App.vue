<template>
  <main style="max-width:720px;margin:40px auto;font:16px/1.5 system-ui">
    <img src="./assets/kamco.gif" alt="Easykam Image" style="max-width: 350px;" />s
    <h1>easyKam</h1>
    <button @click="callApi">Click Me!</button>
    <input v-model="name" placeholder="name" style="margin-left:8px"/>
    <pre v-if="resp" style="background:#111;color:#0f0;padding:8px;border-radius:8px;white-space:pre-wrap">{{ resp }}</pre>
  </main>
</template>

<script setup>
import { ref } from 'vue'
const name = ref('')
const resp = ref('')

async function callApi() {
  const q = name.value ? `?name=${encodeURIComponent(name.value)}` : ''
  const r = await fetch(`/api/hello${q}`)
  resp.value = JSON.stringify(await r.json(), null, 2)
}
</script>
