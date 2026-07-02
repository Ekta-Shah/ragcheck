from ragcheck.cache import JudgmentCache, make_key


def test_make_key_is_stable_and_distinct():
    assert make_key("m", "v1", "q", "c", "a") == make_key("m", "v1", "q", "c", "a")
    assert make_key("m", "v1", "q", "c", "a") != make_key("m", "v2", "q", "c", "a")
    # length-prefixing prevents boundary collisions
    assert make_key("ab", "c") != make_key("a", "bc")


def test_cache_roundtrip_and_counters(tmp_path):
    cache = JudgmentCache(tmp_path / "c.sqlite")
    assert cache.get("k1") is None
    cache.set("k1", "verdict")
    assert cache.get("k1") == "verdict"
    assert cache.hits == 1
    assert cache.misses == 1


def test_cache_persists_across_reopen(tmp_path):
    path = tmp_path / "c.sqlite"
    first = JudgmentCache(path)
    first.set("k", "v")
    first.close()
    second = JudgmentCache(path)
    assert second.get("k") == "v"
