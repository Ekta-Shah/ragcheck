---
version: v1
description: Judge whether a cited chunk supports the sentence citing it.
---
You are verifying citations. The sentence below cites a source document. Decide whether that document actually supports the sentence's factual content. It is SUPPORTED only if the document contains the information the sentence asserts.

Cited document:
{chunk}

Sentence:
{sentence}

Respond with exactly one word: SUPPORTED or UNSUPPORTED.
