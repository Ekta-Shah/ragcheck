---
version: v1
description: Rate how directly an answer addresses the question, 1-5.
---
You are evaluating a question-answering system. Rate how directly the answer addresses the question on this scale:

5 - Fully answers the question, directly and completely.
4 - Answers the question but with minor gaps or slight indirection.
3 - Partially answers; addresses the topic but misses part of what was asked.
2 - Barely relevant; mostly off-target or evasive without justification.
1 - Does not address the question at all.

Note: an explicit, justified refusal (e.g. "the documents do not contain this") on a question the context cannot answer should be rated 5, not penalized.

Question:
{question}

Answer:
{answer}

Respond with ONLY the integer rating.
