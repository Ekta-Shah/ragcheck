---
version: v1
description: Generate one cross-document comparison QA pair (hard tier).
---
Below are excerpts from two different documents. Write ONE question whose answer requires information from BOTH documents - typically a comparison or combination. Provide the short answer.

Rules:
- The question must name both subjects explicitly (e.g. both companies).
- The answer must be derivable strictly from the two excerpts (no outside knowledge).
- Good patterns: "which of X and Y ...", "how do X and Y differ on ...", combining one fact from each.

Excerpt from "{doc_a}":
{text_a}

Excerpt from "{doc_b}":
{text_b}

Respond with ONLY a JSON object: {{"question": "...", "answer": "..."}}
