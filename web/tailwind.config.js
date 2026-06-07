/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        bg: '#0b0f1a',
        surface: '#111827',
        surface2: '#1a2235',
        surface3: '#222d42',
        border: '#2a3655',
        border2: '#3a4a6b',
        primary: '#3b82f6',
        primary2: '#60a5fa',
        primaryDim: 'rgba(59,130,246,.12)',
        accent: '#8b5cf6',
        accentDim: 'rgba(139,92,246,.12)',
        success: '#22c55e',
        successDim: 'rgba(34,197,94,.12)',
        warning: '#f59e0b',
        danger: '#ef4444',
      },
    },
  },
  plugins: [],
}
