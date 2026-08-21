import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  /* config options here */
  eslint: {
    // Warning: This allows production builds to successfully complete even if your project has ESLint errors.
    // NOTE: `next lint` is deprecated in Next 15 and the repo has pre-existing
    // ESLint errors (mostly no-explicit-any) — this gate stays on deliberately.
    ignoreDuringBuilds: true,
  },
  typescript: {
    // Type-checked: `npx tsc --noEmit` is clean, so let build failures surface.
    ignoreBuildErrors: false,
  },
  webpack: (config, { dev }) => {
    if (dev) {
      // The frontend shares its root with the Python backend. The gateway
      // writes backend/chat_data.db on every persisted chat turn; if the dev
      // watcher sees those writes it Fast-Refreshes mid-stream, remounting the
      // page and aborting the in-flight SSE fetch (net::ERR_ABORTED) — which
      // presents as "streaming stopped working". Keep the watcher on frontend
      // sources only.
      config.watchOptions = {
        ...config.watchOptions,
        ignored: [
          "**/node_modules/**",
          "**/.git/**",
          "**/.next/**",
          "**/backend/**",
          "**/rag_system/**",
          "**/eval/**",
          "**/lancedb/**",
          "**/index_store/**",
          "**/shared_uploads/**",
          "**/logs/**",
          "**/.venv/**",
          "**/Documentation/**",
          "**/*.db",
          "**/*.db-wal",
          "**/*.db-shm",
        ],
      };
    }
    return config;
  },
};

export default nextConfig;
