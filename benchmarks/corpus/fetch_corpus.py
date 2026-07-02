"""Fetch SEC 10-K filings from EDGAR and convert to clean text.

Raw data lands in benchmarks/corpus/data/ (gitignored). Usage:

    python benchmarks/corpus/fetch_corpus.py
"""

from __future__ import annotations

import json
import re
import time
from html.parser import HTMLParser
from pathlib import Path
from urllib.request import Request, urlopen

# EDGAR requires a descriptive User-Agent identifying the requester.
USER_AGENT = "ragcheck-benchmark contact@ragcheck.dev"
DATA_DIR = Path(__file__).parent / "data"

COMPANIES = {
    "AAPL": "0000320193",
    "MSFT": "0000789019",
    "NVDA": "0001045810",
}


class _TextExtractor(HTMLParser):
    """Strip tags, skipping script/style, collapsing whitespace."""

    def __init__(self) -> None:
        super().__init__()
        self.chunks: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in ("script", "style"):
            self._skip_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag in ("script", "style") and self._skip_depth:
            self._skip_depth -= 1

    def handle_data(self, data: str) -> None:
        if not self._skip_depth and data.strip():
            self.chunks.append(data)


def _get(url: str) -> bytes:
    request = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(request) as response:
        return bytes(response.read())


def latest_10k_url(cik: str) -> str:
    """Resolve the primary document URL of a company's most recent 10-K."""
    submissions = json.loads(_get(f"https://data.sec.gov/submissions/CIK{cik}.json"))
    recent = submissions["filings"]["recent"]
    for form, accession, doc in zip(
        recent["form"], recent["accessionNumber"], recent["primaryDocument"], strict=True
    ):
        if form == "10-K":
            accession_id = accession.replace("-", "")
            return (
                f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession_id}/{doc}"
            )
    raise LookupError(f"No 10-K found for CIK {cik}")


def html_to_text(html: str) -> str:
    """Extract readable text from filing HTML."""
    extractor = _TextExtractor()
    extractor.feed(html)
    text = "\n".join(extractor.chunks)
    text = text.replace("\xa0", " ")
    return re.sub(r"\n{3,}", "\n\n", text)


def main() -> None:
    """Download the latest 10-K for each company as clean text."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for ticker, cik in COMPANIES.items():
        out = DATA_DIR / f"{ticker}_10K.txt"
        if out.exists():
            print(f"{ticker}: already fetched ({out.stat().st_size // 1024} KB)")
            continue
        url = latest_10k_url(cik)
        print(f"{ticker}: fetching {url}")
        text = html_to_text(_get(url).decode("utf-8", errors="replace"))
        out.write_text(text)
        print(f"{ticker}: wrote {out.stat().st_size // 1024} KB")
        time.sleep(0.5)  # stay well under EDGAR's 10 req/s limit


if __name__ == "__main__":
    main()
