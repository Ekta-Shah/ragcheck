---
version: v1
description: Generate one QA pair requiring synthesis across excerpts of one document (medium tier).
---
Below are consecutive excerpts from the document "{doc}". Write ONE question whose answer requires COMBINING information from at least two of the excerpts - it must not be answerable from any single excerpt alone. Provide the short answer.

Rules:
- The question must be self-contained: name the subject so it has a single correct answer.
- The answer must be derivable strictly from the excerpts (no outside knowledge).
- Good patterns: relating a figure to its context, connecting a policy to its condition, combining two stated facts.

Excerpts:
{text}

Respond with ONLY a JSON object: {{"question": "...", "answer": "..."}}
