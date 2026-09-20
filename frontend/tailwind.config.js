export default {
  darkMode: 'class',
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        background: 'var(--color-bg)',
        card: 'var(--color-card)',
        border: 'var(--color-border)',
        accent: '#FF5A4E', // Coral/Red
        accentGlow: 'rgba(255, 90, 78, 0.4)',
        secondaryAccent: '#8B5CF6', // Warm purple
        textMain: 'var(--color-text-main)',
        textMuted: 'var(--color-text-muted)',
        inputBg: 'var(--color-input-bg)',
        inputBorder: 'var(--color-input-border)',
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
        display: ['Inter', 'system-ui', 'sans-serif'], // Fallback for Cabinet Grotesk
      },
      boxShadow: {
        'card-light': '0 8px 30px rgba(0,0,0,0.06)',
        'card-dark-ambient': '0 0 40px rgba(255,90,78,0.05)',
        'btn-glow': '0 0 24px rgba(239,68,68,0.4)',
        'input-focus': '0 0 0 3px rgba(255,90,78,0.15)',
        'verdict-approved': '0 0 40px rgba(34,197,94,0.15)',
        'verdict-rejected': '0 0 40px rgba(239,68,68,0.15)',
      }
    },
  },
  plugins: [],
}
