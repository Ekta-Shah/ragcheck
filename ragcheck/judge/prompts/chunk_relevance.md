---
version: v1
description: Judge whether a retrieved chunk is relevant to answering a question.
---
You are evaluating a retrieval system. Decide whether the following document chunk contains information that helps answer the question. A chunk is RELEVANT only if it contains facts needed for the answer; topical similarity alone is not enough.

Question:
{question}

Chunk:
{chunk}

Respond with exactly one word: RELEVANT or IRRELEVANT.
