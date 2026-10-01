"""InflightStore: the backend handles of jobs whose work is in flight."""

from __future__ import annotations

from vorqd.state import InflightRow, InflightStore


def test_put_get_delete_roundtrip():
    store = InflightStore(None, clock=lambda: 1234.0)
    assert store.get("job_a") is None
    store.put("job_a", "m:fp8", "resp_1")
    assert store.get("job_a") == InflightRow("job_a", "m:fp8", "resp_1", 1234.0)
    store.delete("job_a")
    assert store.get("job_a") is None
    store.delete("job_a")              # deleting nothing is not an error


def test_put_overwrites_the_handle_of_a_resubmitted_job():
    store = InflightStore(None)
    store.put("job_a", "m:fp8", "resp_1")
    store.put("job_a", "m:fp8", "resp_2")
    assert store.get("job_a").handle == "resp_2"
    assert len(store.all()) == 1


def test_all_lists_rows_oldest_first():
    ticks = iter([10.0, 5.0, 7.0])
    store = InflightStore(None, clock=lambda: next(ticks))
    store.put("late", "m", "h1")
    store.put("early", "m", "h2")
    store.put("mid", "m", "h3")
    assert [r.job_id for r in store.all()] == ["early", "mid", "late"]


def test_a_file_store_survives_reopening(tmp_path):
    path = str(tmp_path / "state.sqlite")
    first = InflightStore(path)
    first.put("job_a", "m:fp8", "resp_1")
    first.close()

    second = InflightStore(path)
    assert second.get("job_a").handle == "resp_1"
    second.close()
