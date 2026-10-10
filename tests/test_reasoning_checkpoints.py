from pathlib import Path
from typing import Any

import pytest

from nanochat import checkpoint_manager as manager


@pytest.mark.parametrize("source,directory", [
    ("base", "base_checkpoints"), ("sft", "chatsft_checkpoints"),
    ("rl", "chatrl_checkpoints"), ("reason_sft", "reasonsft_checkpoints"),
    ("reason_rl", "reasonrl_checkpoints"),
])
def test_model_and_optimizer_use_same_source_mapping(monkeypatch: Any, tmp_path: Path, source: str, directory: str) -> None:
    monkeypatch.setattr(manager, "get_base_dir", lambda: str(tmp_path))
    monkeypatch.setattr(manager, "load_model_from_dir", lambda root, **kwargs: root)
    assert manager.load_model(source) == str(tmp_path / directory)
    checkpoint = tmp_path / directory / "d24" / "optim_000486_rank0.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"fixture")
    monkeypatch.setattr(manager.torch, "load", lambda path, **kwargs: path)
    assert manager.load_optimizer_state(source, "cpu", 0, "d24", 486) == str(checkpoint)
