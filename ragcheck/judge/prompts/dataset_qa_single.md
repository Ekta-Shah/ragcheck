---
version: v1
description: Generate one single-chunk factual QA pair (easy tier).
---
Below is an excerpt from the document "{doc}". Write ONE factual question that is fully answered by this excerpt alone, plus its short answer.

Rules:
- The question must be self-contained: name the subject (company, entity, document) so it has a single correct answer without seeing the excerpt.
- The answer must be a short fact stated in the excerpt (a number, name, date, or short phrase).
- Prefer substantive facts over boilerplate; if the excerpt is mostly boilerplate, pick the most concrete detail available.

Excerpt:
{text}

Respond with ONLY a JSON object: {{"question": "...", "answer": "..."}}
