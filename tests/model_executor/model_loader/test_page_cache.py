import os

from vllm.model_executor.model_loader import weight_utils


def test_safetensors_load_device_defaults_to_cpu(monkeypatch) -> None:
    monkeypatch.delenv("SAFETENSORS_LOAD_DEVICE", raising=False)

    assert weight_utils._safetensors_load_device() == "cpu"


def test_safetensors_load_device_uses_override(monkeypatch) -> None:
    monkeypatch.setenv("SAFETENSORS_LOAD_DEVICE", "cuda:0")

    assert weight_utils._safetensors_load_device() == "cuda:0"


def test_safetensors_load_cache_release_skips_cpu(monkeypatch) -> None:
    calls = []
    monkeypatch.delenv("SAFETENSORS_LOAD_DEVICE", raising=False)
    monkeypatch.setattr(
        weight_utils.torch.accelerator,
        "empty_cache",
        lambda: calls.append("empty"),
    )

    weight_utils._release_safetensors_load_cache()

    assert calls == []


def test_safetensors_load_cache_release_synchronizes_device(monkeypatch) -> None:
    calls = []
    monkeypatch.setenv("SAFETENSORS_LOAD_DEVICE", "cuda:0")
    monkeypatch.setattr(
        weight_utils.torch.accelerator,
        "synchronize",
        lambda device: calls.append(("synchronize", device)),
    )
    monkeypatch.setattr(
        weight_utils.torch.accelerator,
        "empty_cache",
        lambda: calls.append(("empty", None)),
    )

    weight_utils._release_safetensors_load_cache()

    assert calls == [
        ("synchronize", weight_utils.torch.device("cuda:0")),
        ("empty", None),
    ]


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
