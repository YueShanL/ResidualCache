from types import SimpleNamespace

import pytest


torch = pytest.importorskip("torch")

from output_preserving_region_router.config import (  # noqa: E402
    OutputPreservationLossConfig,
    RegionRouterConfig,
)
from output_preserving_region_router.gated_kv_adapter import (  # noqa: E402
    Gemma4SoftBlockGateController,
)
from output_preserving_region_router.losses import output_preservation_loss  # noqa: E402
from output_preserving_region_router.model import GaussianRegionRouter  # noqa: E402
from output_preserving_region_router.trainer import (  # noqa: E402
    _checkpoint_selection_score,
)


def _attention_module():
    return SimpleNamespace(
        layer_idx=0,
        layer_type="full_attention",
        is_kv_shared_layer=False,
        config=SimpleNamespace(
            num_hidden_layers=1,
            num_kv_shared_layers=0,
            layer_types=["full_attention"],
        ),
        num_key_value_groups=1,
        head_dim=2,
        training=False,
    )


def test_gaussian_router_produces_query_conditioned_region_and_hard_mask():
    config = RegionRouterConfig(
        residual_dim=3,
        feature_dim=2,
        hidden_dim=4,
        depth=1,
        dropout=0.0,
        minimum_scale=0.1,
        radius=2.0,
        gate_temperature=0.2,
    )
    model = GaussianRegionRouter(config)
    query_mean = torch.zeros(1, 2)
    query_scale = torch.ones(1, 2)
    keys = torch.tensor([[[0.0, 0.0], [2.0, 0.0], [5.0, 0.0]]])
    mask = torch.tensor([[True, True, False]])

    output = model.score_features(query_mean, query_scale, keys, mask)

    assert torch.allclose(output.squared_distances[0, :2], torch.tensor([0.0, 0.5]))
    assert output.hard_mask.tolist() == [[True, True, False]]
    assert output.gates[0, 0] > output.gates[0, 1]
    assert output.gates[0, 2] == 0


def test_output_objective_penalizes_actual_gate_count_and_backpropagates():
    full = torch.tensor([[[2.0, 0.0], [0.0, 2.0]]])
    student = torch.tensor(
        [[[1.5, 0.5], [0.5, 1.5]]], requires_grad=True
    )
    gates = torch.tensor([[0.8, 0.2, 0.0]], requires_grad=True)
    mask = torch.tensor([[True, True, False]])
    loss = output_preservation_loss(
        full,
        student,
        full,
        gates,
        mask,
        OutputPreservationLossConfig(
            maximum_excess_output_kl=0.0,
            preservation_weight=2.0,
            sparsity_weight=1.0,
            gate_entropy_weight=0.0,
        ),
    )

    assert torch.allclose(loss.expected_selected_blocks, torch.tensor(1.0))
    assert torch.allclose(loss.expected_selected_fraction, torch.tensor(0.5))
    loss.total.backward()
    assert student.grad is not None and torch.isfinite(student.grad).all()
    assert gates.grad is not None and torch.isfinite(gates.grad).all()
    assert gates.grad[0, 2] == 0


def test_soft_block_gate_changes_attention_and_preserves_router_gradient():
    historical_key = torch.zeros(1, 1, 2, 2)
    historical_value = torch.tensor([[[[2.0, 0.0], [0.0, 2.0]]]])
    gate_logits = torch.tensor([[2.0, -2.0]], requires_grad=True)
    gates = torch.sigmoid(gate_logits)
    controller = Gemma4SoftBlockGateController(
        {0: (historical_key, historical_value)},
        gates,
        (1, 1),
        gate_epsilon=1e-6,
    )
    query = torch.zeros(1, 1, 1, 2)
    native_key = torch.zeros(1, 1, 1, 2)
    native_value = torch.zeros(1, 1, 1, 2)

    output, probabilities = controller.attend(
        _attention_module(),
        query,
        native_key,
        native_value,
        None,
        dropout=0.0,
        scaling=1.0,
        softcap=None,
    )

    assert probabilities.shape == (1, 1, 1, 3)
    assert probabilities[0, 0, 0, 0] > probabilities[0, 0, 0, 1]
    output.square().sum().backward()
    assert gate_logits.grad is not None
    assert torch.isfinite(gate_logits.grad).all()
    assert gate_logits.grad.abs().sum() > 0


def test_soft_block_gate_keeps_gradient_at_zero_gate():
    historical_key = torch.zeros(1, 1, 2, 2)
    historical_value = torch.tensor([[[[2.0, 0.0], [0.0, 2.0]]]])
    gates = torch.tensor([[0.0, 1.0]], requires_grad=True)
    controller = Gemma4SoftBlockGateController(
        {0: (historical_key, historical_value)},
        gates,
        (1, 1),
        gate_epsilon=1e-6,
    )

    output, _probabilities = controller.attend(
        _attention_module(),
        torch.zeros(1, 1, 1, 2),
        torch.zeros(1, 1, 1, 2),
        torch.zeros(1, 1, 1, 2),
        None,
        dropout=0.0,
        scaling=1.0,
        softcap=None,
    )
    output.square().sum().backward()

    assert gates.grad is not None
    assert torch.isfinite(gates.grad).all()
    assert gates.grad[0, 0].abs() > 0


def test_checkpoint_selection_never_trades_constraint_for_one_fewer_block():
    feasible = {
        "output_kl_violation": 0.0,
        "expected_selected_blocks": 8.0,
        "output_kl": 0.2,
    }
    infeasible_but_smaller = {
        "output_kl_violation": 0.001,
        "expected_selected_blocks": 1.0,
        "output_kl": 0.01,
    }

    assert _checkpoint_selection_score(feasible) < _checkpoint_selection_score(
        infeasible_but_smaller
    )
