"""Exercise the real SFT loop with local data and a tiny differentiable model."""

from types import SimpleNamespace
from typing import Any

import pytest
import torch


@pytest.mark.parametrize("horizon,updates", [(3, 3), (-1, 2)])
def test_sft_explicit_horizon_fresh_optimizer_and_packing(monkeypatch: Any, tmp_path: Any, horizon: int, updates: int) -> None:
    from scripts import reason_sft

    class Tokenizer:
        def get_bos_token_id(self) -> int:
            return 0

        def render_conversation(self, conversation: dict[str, Any], max_tokens: Any = 2048) -> tuple[list[int], list[int]]:
            assert max_tokens is None
            return conversation["ids"], conversation["mask"]

    class TinyModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.logits = torch.nn.Parameter(torch.zeros(6))
            self.config = SimpleNamespace(sequence_len=16, n_layer=1)
            self.updates = 0

        def setup_optimizer(self, **kwargs: Any) -> Any:
            optimizer = torch.optim.SGD([{"params": self.parameters(), "kind": "adamw"}], lr=0.1)
            original_step = optimizer.step

            def step() -> Any:
                assert all(group["lr"] >= 0 for group in optimizer.param_groups)
                self.updates += 1
                return original_step()

            optimizer.step = step
            return optimizer

        def estimate_flops(self) -> int:
            return 1

        def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
            assert (targets >= 0).any()
            assert not (targets == 4).any()  # Tool-output token must stay masked.
            return torch.nn.functional.cross_entropy(
                self.logits.expand(*targets.shape, 6).reshape(-1, 6),
                targets.reshape(-1), ignore_index=-1,
            )

    model = TinyModel()
    dataset = [{"ids": [0, 1, 2, 3, 4, 5], "mask": [0, 0, 0, 1, 0, 1]}] * 8
    monkeypatch.setattr(reason_sft, "compute_init", lambda _: (False, 0, 0, 1, torch.device("cpu")))
    monkeypatch.setattr(reason_sft, "compute_cleanup", lambda: None)
    monkeypatch.setattr(reason_sft, "load_model", lambda *a, **k: (model, Tokenizer(), {"step": 486}))
    monkeypatch.setattr(reason_sft, "get_base_dir", lambda: str(tmp_path))
    monkeypatch.setattr(reason_sft, "get_token_bytes", lambda **k: torch.ones(6))
    monkeypatch.setattr(reason_sft, "build_sft_mixture", lambda *a, **k: (dataset, dataset))
    monkeypatch.setattr(reason_sft, "evaluate_bpb", lambda *a: 0.5)
    monkeypatch.setattr(reason_sft.gc, "freeze", lambda: None)
    monkeypatch.setattr(reason_sft.gc, "disable", lambda: None)
    saved: dict[str, Any] = {}
    monkeypatch.setattr(reason_sft, "save_checkpoint", lambda directory, step, state, optimizer, metadata, **k: saved.update(metadata))
    reason_sft.main([
        "--device-type", "cpu", "--model-tag", "d24", "--model-step", "486",
        "--output-tag", "tiny-test", "--num-iterations", str(horizon), "--no-compile",
        "--device-batch-size", "2", "--total-batch-size", "32", "--chatcore-every", "-1",
    ])
    assert model.updates == saved["step"] == updates
    assert saved["parent"] == {"source": "sft", "model_tag": "d24", "step": 486}
    assert saved["model_config"]["sequence_len"] == 16
    assert saved["user_config"]["max_seq_len"] == 16
    assert len((tmp_path / "reasonsft_checkpoints" / "tiny-test" / "training.jsonl").read_text().splitlines()) == updates
