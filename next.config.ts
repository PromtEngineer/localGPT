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
};

export default nextConfig;
