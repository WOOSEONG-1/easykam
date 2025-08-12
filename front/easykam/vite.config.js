import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

// https://vite.dev/config/
export default defineConfig({
  plugins: [vue()],
  server: {
    proxy: {
      '/api': {
        target: 'https://easykam.life', // 로컬 개발 시 서버 API로 프록시
        changeOrigin: true
      }
    }
  }
})
