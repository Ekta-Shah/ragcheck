---
version: v1
description: Detect whether a response is a refusal / "I don't know" rather than an answer.
---
You are classifying a question-answering system's response. Decide whether the response ANSWERS the question or REFUSES it. A response REFUSES when it says the information is unavailable, not in the provided documents, unknown, or otherwise declines to give a substantive answer. A response that gives a substantive answer (even a wrong or partial one) counts as ANSWER.

Question:
{question}

Response:
{answer}

Respond with exactly one word: ANSWER or REFUSAL.
