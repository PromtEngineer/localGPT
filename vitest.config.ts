import { resolve } from 'node:path'
import react from '@vitejs/plugin-react'
import { defineConfig } from 'vitest/config'

// Minimal component-test setup (jsdom + Testing Library). Scoped to UI
// render smokes — keep these fast and dependency-light.
export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: { '@': resolve(process.cwd(), 'src') },
  },
  // Override the project's Tailwind v4 PostCSS config — render smokes don't
  // need CSS processing and the prod config fails to load under vitest.
  css: { postcss: { plugins: [] } },
  test: {
    environment: 'jsdom',
    globals: true,
    setupFiles: ['./src/test/setup.ts'],
    include: ['src/**/*.test.{ts,tsx}'],
    css: false,
  },
})
