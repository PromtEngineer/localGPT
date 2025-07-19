
const testMarkdownWithExcessiveNewlines = `# Test Response

This is a test response with excessive newlines.



Here's some content after multiple empty lines.




## Section Header

More content here.






### Subsection

Final content with lots of spacing.




The end.`;

const testStreamingTokens = [
  "# Test Response\n\n",
  "This is a test response",
  " with excessive newlines.\n\n\n\n",
  "Here's some content after",
  " multiple empty lines.\n\n\n\n\n",
  "## Section Header\n\n",
  "More content here.\n\n\n\n\n\n\n",
  "### Subsection\n\n",
  "Final content with lots",
  " of spacing.\n\n\n\n\n",
  "The end."
];

function currentCleanup(text) {
  return text.replace(/\n{3,}/g, '\n\n');
}

function improvedCleanup(text) {
  text = text.replace(/\n{3,}/g, '\n\n');
  
  text = text.replace(/[ \t]+$/gm, '');
  
  text = text.replace(/[ \t]{3,}/g, ' ');
  
  text = text.replace(/[ \t]*\n[ \t]*\n[ \t]*\n/g, '\n\n');
  
  text = text.trim();
  
  return text;
}

console.log("=== ORIGINAL TEXT ===");
console.log(JSON.stringify(testMarkdownWithExcessiveNewlines));

console.log("\n=== CURRENT CLEANUP ===");
console.log(JSON.stringify(currentCleanup(testMarkdownWithExcessiveNewlines)));

console.log("\n=== IMPROVED CLEANUP ===");
console.log(JSON.stringify(improvedCleanup(testMarkdownWithExcessiveNewlines)));

console.log("\n=== STREAMING SIMULATION ===");
let streamedText = "";
testStreamingTokens.forEach((token, i) => {
  streamedText += token;
  console.log(`Token ${i + 1}: "${token}"`);
  console.log(`Accumulated (current): "${currentCleanup(streamedText)}"`);
  console.log(`Accumulated (improved): "${improvedCleanup(streamedText)}"`);
  console.log("---");
});
