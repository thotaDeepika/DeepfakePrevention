import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    host: '0.0.0.0', // Listen on all IPs for phone testing
    proxy: {
      '/protect': {
        target: 'http://127.0.0.1:7860',
        changeOrigin: true,
      },
      '/download': {
        target: 'http://127.0.0.1:7860',
        changeOrigin: true,
      },
      '/gemini_eval': {
        target: 'http://127.0.0.1:7860',
        changeOrigin: true,
      }
    }
  }
})
