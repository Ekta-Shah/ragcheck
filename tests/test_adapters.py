from ragcheck.adapters import FunctionAdapter, RAGResponse, RetrievedChunk


def make_response(question: str) -> RAGResponse:
    return RAGResponse(
        answer=f"answer to {question}",
        retrieved_chunks=[RetrievedChunk(content="ctx", source_id="doc1", score=0.9)],
        latencies_ms={"retrieval": 5.0},
        token_usage={"input_tokens": 100},
    )


def test_function_adapter_query():
    adapter = FunctionAdapter(make_response)
    response = adapter.query("q1")
    assert response.answer == "answer to q1"
    assert response.retrieved_chunks[0].source_id == "doc1"
    assert response.refused is False


def test_function_adapter_batch_query_is_sequential():
    adapter = FunctionAdapter(make_response)
    responses = adapter.batch_query(["q1", "q2", "q3"])
    assert [r.answer for r in responses] == ["answer to q1", "answer to q2", "answer to q3"]
