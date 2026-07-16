import { dirname } from "path";
import { fileURLToPath } from "url";
import { FlatCompat } from "@eslint/eslintrc";

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const compat = new FlatCompat({
  baseDirectory: __dirname,
});

const eslintConfig = [
  { ignores: [".next/**", "next-env.d.ts"] },
  ...compat.extends("next/core-web-vitals", "next/typescript"),
  {
    rules: {
      // The existing UI has several dynamic renderer boundaries. Type checking
      // remains strict, while incremental replacement of these escape hatches
      // is tracked as lint warnings rather than blocking secure builds.
      "@typescript-eslint/no-explicit-any": "warn",
    },
  },
];

export default eslintConfig;
