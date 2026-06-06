import { dirname } from "path";
import { fileURLToPath } from "url";
import { FlatCompat } from "@eslint/eslintrc";

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const compat = new FlatCompat({
  baseDirectory: __dirname,
});

const eslintConfig = [
  {
    ignores: [
      ".next/**",
      "node_modules/**",
      ".venv/**",
      "lancedb/**",
      "index_store/**",
      "shared_uploads/**",
      "logs/**",
      "coverage/**",
      "out/**",
      "build/**",
      "__pycache__/**",
      ".pytest_cache/**",
    ],
  },
  ...compat.extends("next/core-web-vitals", "next/typescript"),
];

export default eslintConfig;
