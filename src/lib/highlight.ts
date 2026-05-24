import * as React from "react";

const STOP_WORDS = new Set([
  "a","an","the","is","are","was","were","be","been","being","have","has","had",
  "do","does","did","will","would","shall","should","may","might","must","can","could",
  "to","of","in","on","at","by","for","with","as","from","up","about","into","or","and",
  "not","it","its","this","that","these","those","what","which","who","whom","how","when","where","why",
]);

function escapeRegex(str: string): string {
  return str.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

/**
 * Splits `text` by terms extracted from `query`, returning a React fragment
 * where matched terms are wrapped in <mark> elements for highlighting.
 */
export function highlightTerms(text: string, query: string): React.ReactNode {
  const tokens = query
    .split(/\s+/)
    .map((t) => t.replace(/[^a-zA-Z0-9]/g, "").toLowerCase())
    .filter((t) => t.length >= 3 && !STOP_WORDS.has(t));

  if (tokens.length === 0) return text;

  const pattern = new RegExp(`(${tokens.map(escapeRegex).join("|")})`, "gi");
  const parts = text.split(pattern);

  return React.createElement(
    React.Fragment,
    null,
    ...parts.map((part, i) =>
      pattern.test(part)
        ? React.createElement("mark", { key: i, className: "bg-yellow-400/30 text-inherit rounded-sm" }, part)
        : part,
    ),
  );
}
