from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from .contracts import Vector


@dataclass
class LearnableRouterEncoder:
    """Optional boundary adapter for an existing learnable-index checkpoint.

    Imports from :mod:`learnable_index` are deliberately lazy.  The learned
    index package remains unaware of this bridge and can still run standalone.
    """

    model: Any
    device: Any

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: str | Path,
        *,
        device: str = "cpu",
    ) -> "LearnableRouterEncoder":
        try:
            import torch
            from learnable_index.trainer import load_checkpoint
        except ImportError as error:  # pragma: no cover - runtime dependency guard.
            raise RuntimeError("loading a learned router requires torch and learnable_index") from error
        model, _router_config, _loss_config, _train_config, _payload = load_checkpoint(
            Path(checkpoint), map_location="cpu"
        )
        model = model.to(torch.device(device))
        model.eval()
        return cls(model=model, device=torch.device(device))

    @staticmethod
    def _as_batch(values: Any, *, device: Any):
        import torch

        tensor = torch.as_tensor(values, dtype=torch.float32, device=device)
        if tensor.ndim != 1:
            raise ValueError("residual summary must be one-dimensional")
        return tensor.unsqueeze(0)

    def _encode(self, tower: Any, residual_summary: Sequence[float] | Any) -> Vector:
        encoded = self._encode_tensor(tower, residual_summary)
        return tuple(float(value) for value in encoded.cpu())

    def _encode_tensor(self, tower: Any, residual_summary: Sequence[float] | Any):
        import torch

        with torch.no_grad():
            encoded = tower(self._as_batch(residual_summary, device=self.device))[0]
        return encoded.detach().float()

    def encode_block(self, residual_summary: Sequence[float] | Any) -> Vector:
        return self._encode(self.model.key_network, residual_summary)

    def encode_query(self, residual_summary: Sequence[float] | Any) -> Vector:
        return self._encode(self.model.query_network, residual_summary)

    def encode_block_tensor(self, residual_summary: Sequence[float] | Any):
        """Return the block key on the configured router device."""

        return self._encode_tensor(self.model.key_network, residual_summary)

    def encode_query_tensor(self, residual_summary: Sequence[float] | Any):
        """Return the query key on the configured router device."""

        return self._encode_tensor(self.model.query_network, residual_summary)
