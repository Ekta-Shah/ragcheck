---
version: v1
description: Generate a plausible on-topic question that the corpus cannot answer.
---
Below is an excerpt from the document "{doc}". Write ONE question that is plausible and on-topic for this kind of document, but whose answer is NOT contained in the excerpt and is unlikely to appear anywhere in the document.

Rules:
- Stay on-topic: the question should sound like something a reader of this document would ask.
- Make it specific enough that it cannot be answered by generalities (e.g. ask for a figure, name, or detail the document would not disclose).
- Do NOT ask about anything actually stated in the excerpt.

Excerpt:
{text}

Respond with ONLY a JSON object: {{"question": "..."}}
