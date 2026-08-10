import os

from vllm.model_executor.model_loader import weight_utils


def test_page_cache_eviction_is_opt_in(tmp_path, monkeypatch) -> None:
    path = tmp_path / "weights.safetensors"
    path.write_bytes(b"weights")
    calls = []
    monkeypatch.delenv("SAFETENSORS_DROP_PAGE_CACHE", raising=False)
    monkeypatch.setattr(os, "posix_fadvise", lambda *args: calls.append(args))

    weight_utils._drop_safetensors_page_cache(str(path))

    assert calls == []


def test_page_cache_eviction_advises_dontneed(tmp_path, monkeypatch) -> None:
    path = tmp_path / "weights.safetensors"
    path.write_bytes(b"weights")
    calls = []
    monkeypatch.setenv("SAFETENSORS_DROP_PAGE_CACHE", "1")
    monkeypatch.setattr(os, "posix_fadvise", lambda *args: calls.append(args))

    weight_utils._drop_safetensors_page_cache(str(path))

    assert len(calls) == 1
    assert calls[0][1:] == (0, 0, os.POSIX_FADV_DONTNEED)
