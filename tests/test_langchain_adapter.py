"""LangChain adapter tests with duck-typed fakes (no langchain dependency)."""

from ragcheck.adapters.langchain import LangChainAdapter


class FakeDocument:
    def __init__(self, page_content, metadata=None):
        self.page_content = page_content
        self.metadata = metadata or {}


class FakeRetriever:
    def invoke(self, question):
        return [
            FakeDocument("chunk about leave", {"source": "doc_leave"}),
            FakeDocument("chunk without source metadata"),
        ]


class FakeChain:
    def __init__(self, result):
        self.result = result
        self.last_input = None

    def invoke(self, inputs):
        self.last_input = inputs
        return self.result


class FakeAIMessage:
    def __init__(self, content):
        self.content = content


def test_langchain_adapter_maps_documents_and_answer():
    chain = FakeChain("24 days of leave.")
    adapter = LangChainAdapter(FakeRetriever(), chain)
    response = adapter.query("How much leave?")

    assert response.answer == "24 days of leave."
    assert [c.source_id for c in response.retrieved_chunks] == ["doc_leave", "doc_1"]
    assert response.retrieved_chunks[0].content == "chunk about leave"
    assert chain.last_input["question"] == "How much leave?"
    assert "[doc_leave] chunk about leave" in chain.last_input["context"]
    assert set(response.latencies_ms) == {"retrieval", "generation"}


def test_langchain_adapter_unwraps_message_content():
    adapter = LangChainAdapter(FakeRetriever(), FakeChain(FakeAIMessage("From a message.")))
    assert adapter.query("q").answer == "From a message."
