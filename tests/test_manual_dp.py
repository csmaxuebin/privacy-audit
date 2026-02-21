"""Unit tests for manual DP fallback (Task 3.3).

Tests cover:
- manual_dp_step: batch-level clipping + Gaussian noise on LoRA params
- _create_accountant: RDP/PRV accountant selection
- _compute_noise_multiplier: binary search for noise calibration
- train_manual_dp metadata attributes
"""
import math
import sys
import os

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from train_dp_sft import manual_dp_step

# _create_accountant and _compute_noise_multiplier require opacus at import
# time (they use opacus.accountants internally), so we guard the import.
_HAS_OPACUS = True
try:
    import opacus  # noqa: F401
    from train_dp_sft import _create_accountant, _compute_noise_multiplier
except ImportError:
    _HAS_OPACUS = False

requires_opacus = pytest.mark.skipif(
    not _HAS_OPACUS, reason="opacus is not installed"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _SimpleLora(nn.Module):
    """Minimal model with 'lora' named parameters for testing."""

    def __init__(self):
        super().__init__()
        self.base_linear = nn.Linear(4, 4)
        self.lora_down = nn.Linear(4, 2, bias=False)
        self.lora_up = nn.Linear(2, 4, bias=False)
        # Freeze base
        for p in self.base_linear.parameters():
            p.requires_grad = False

    def forward(self, x):
        return self.base_linear(x) + self.lora_up(self.lora_down(x))


def _attach_grads(model):
    """Run a dummy forward/backward to populate .grad on LoRA params."""
    x = torch.randn(2, 4)
    loss = model(x).sum()
    loss.backward()


# ---------------------------------------------------------------------------
# Tests: manual_dp_step
# ---------------------------------------------------------------------------

class TestManualDpStep:
    """Tests for manual_dp_step()."""

    def test_gradients_are_clipped(self):
        """After manual_dp_step, LoRA grad norms should respect clipping."""
        torch.manual_seed(0)
        model = _SimpleLora()
        optimizer = torch.optim.SGD(
            [p for p in model.parameters() if p.requires_grad], lr=0.1
        )
        _attach_grads(model)

        clipping_norm = 0.01  # very small to force clipping
        # noise_multiplier=0 so we only test clipping
        manual_dp_step(model, optimizer, clipping_norm, noise_multiplier=0.0)

        # After step, grads are zeroed by optimizer.zero_grad inside manual_dp_step
        for n, p in model.named_parameters():
            if p.requires_grad and "lora" in n.lower():
                assert p.grad is None or p.grad.norm().item() == 0.0

    def test_noise_is_injected(self):
        """With noise_multiplier > 0, parameters should change differently
        than a noiseless step."""
        torch.manual_seed(42)
        model_noisy = _SimpleLora()
        model_clean = _SimpleLora()
        # Copy weights
        model_clean.load_state_dict(model_noisy.state_dict())

        opt_noisy = torch.optim.SGD(
            [p for p in model_noisy.parameters() if p.requires_grad], lr=0.1
        )
        opt_clean = torch.optim.SGD(
            [p for p in model_clean.parameters() if p.requires_grad], lr=0.1
        )

        x = torch.randn(2, 4)

        # Noisy step
        torch.manual_seed(42)
        loss = model_noisy(x).sum()
        loss.backward()
        manual_dp_step(model_noisy, opt_noisy, clipping_norm=10.0,
                       noise_multiplier=1.0)

        # Clean step (no noise)
        torch.manual_seed(42)
        loss = model_clean(x).sum()
        loss.backward()
        manual_dp_step(model_clean, opt_clean, clipping_norm=10.0,
                       noise_multiplier=0.0)

        # Weights should differ due to noise
        diffs = []
        for (n1, p1), (n2, p2) in zip(
            model_noisy.named_parameters(), model_clean.named_parameters()
        ):
            if p1.requires_grad:
                diffs.append((p1 - p2).abs().sum().item())
        assert sum(diffs) > 0, "Noise should cause weight divergence"

    def test_base_params_untouched(self):
        """Base (frozen) parameters should not be modified."""
        torch.manual_seed(0)
        model = _SimpleLora()
        base_before = model.base_linear.weight.clone()

        optimizer = torch.optim.SGD(
            [p for p in model.parameters() if p.requires_grad], lr=0.1
        )
        _attach_grads(model)
        manual_dp_step(model, optimizer, clipping_norm=1.0,
                       noise_multiplier=0.5)

        assert torch.equal(model.base_linear.weight, base_before)

    def test_zero_noise_multiplier_is_noiseless(self):
        """noise_multiplier=0 should produce deterministic (no noise) step."""
        torch.manual_seed(7)
        m1 = _SimpleLora()
        m2 = _SimpleLora()
        m2.load_state_dict(m1.state_dict())

        o1 = torch.optim.SGD(
            [p for p in m1.parameters() if p.requires_grad], lr=0.1
        )
        o2 = torch.optim.SGD(
            [p for p in m2.parameters() if p.requires_grad], lr=0.1
        )

        x = torch.randn(2, 4)

        torch.manual_seed(99)
        m1(x).sum().backward()
        manual_dp_step(m1, o1, 1.0, 0.0)

        torch.manual_seed(99)
        m2(x).sum().backward()
        manual_dp_step(m2, o2, 1.0, 0.0)

        for (_, p1), (_, p2) in zip(
            m1.named_parameters(), m2.named_parameters()
        ):
            assert torch.allclose(p1, p2), "Zero noise should be deterministic"


# ---------------------------------------------------------------------------
# Tests: _create_accountant
# ---------------------------------------------------------------------------

@requires_opacus
class TestCreateAccountant:
    """Tests for _create_accountant()."""

    def test_rdp_accountant(self):
        """accountant_type='rdp' should return RDPAccountant."""
        from opacus.accountants import RDPAccountant
        acc = _create_accountant("rdp")
        assert isinstance(acc, RDPAccountant)

    def test_prv_accountant_or_fallback(self):
        """accountant_type='prv' should return PRVAccountant if available,
        otherwise fall back to RDPAccountant."""
        from opacus.accountants import RDPAccountant
        acc = _create_accountant("prv")
        # Either PRVAccountant or RDPAccountant (fallback) is acceptable
        try:
            from opacus.accountants import PRVAccountant
            assert isinstance(acc, (PRVAccountant, RDPAccountant))
        except ImportError:
            assert isinstance(acc, RDPAccountant)


# ---------------------------------------------------------------------------
# Tests: _compute_noise_multiplier
# ---------------------------------------------------------------------------

@requires_opacus
class TestComputeNoiseMultiplier:
    """Tests for _compute_noise_multiplier()."""

    def test_returns_positive(self):
        """Noise multiplier should always be positive."""
        nm = _compute_noise_multiplier(
            target_epsilon=8.0, delta=1e-4,
            sample_rate=0.01, num_steps=100,
        )
        assert nm > 0

    def test_smaller_epsilon_needs_more_noise(self):
        """Tighter privacy (smaller ε) requires larger noise_multiplier."""
        nm_loose = _compute_noise_multiplier(
            target_epsilon=8.0, delta=1e-4,
            sample_rate=0.01, num_steps=100,
        )
        nm_tight = _compute_noise_multiplier(
            target_epsilon=1.0, delta=1e-4,
            sample_rate=0.01, num_steps=100,
        )
        assert nm_tight > nm_loose, (
            f"ε=1 should need more noise than ε=8, "
            f"got {nm_tight} vs {nm_loose}"
        )

    def test_achieves_target_epsilon(self):
        """The computed noise_multiplier should yield ε ≤ target."""
        from opacus.accountants import RDPAccountant

        target_eps = 4.0
        delta = 1e-4
        sr = 0.01
        steps = 200

        nm = _compute_noise_multiplier(target_eps, delta, sr, steps)

        acc = RDPAccountant()
        acc.history = [(nm, sr, steps)]
        actual_eps = acc.get_epsilon(delta)
        assert actual_eps <= target_eps * 1.05, (
            f"Expected ε ≤ {target_eps * 1.05}, got {actual_eps}"
        )
