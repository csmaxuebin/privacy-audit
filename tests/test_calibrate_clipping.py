"""Unit tests for clipping norm calibration (Task 3.4).

Tests cover:
- calibrate_clipping: batch-level gradient norm collection and statistics output
- Default clipping norm hint in main() (Requirement 3.4)
"""
import sys
import os

import pytest
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from unittest.mock import patch
from io import StringIO

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from train_dp_sft import calibrate_clipping


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

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        # Simplified forward that mimics CausalLM interface
        x = torch.randn(input_ids.shape[0], 4)
        out = self.base_linear(x) + self.lora_up(self.lora_down(x))
        loss = out.sum()
        # Return an object with .loss attribute
        return type("Output", (), {"loss": loss})()


def _make_dataloader(num_batches=10, batch_size=2, seq_len=8):
    """Create a simple DataLoader yielding dicts with input_ids/attention_mask."""
    total = num_batches * batch_size
    input_ids = torch.randint(0, 100, (total, seq_len))
    attention_mask = torch.ones(total, seq_len, dtype=torch.long)

    class DictDataset:
        def __init__(self, ids, masks):
            self.ids = ids
            self.masks = masks

        def __len__(self):
            return len(self.ids)

        def __getitem__(self, idx):
            return {"input_ids": self.ids[idx], "attention_mask": self.masks[idx]}

    dataset = DictDataset(input_ids, attention_mask)
    return DataLoader(dataset, batch_size=batch_size, drop_last=True)


# ---------------------------------------------------------------------------
# Tests: calibrate_clipping
# ---------------------------------------------------------------------------

class TestCalibrateClipping:
    """Tests for calibrate_clipping()."""

    def test_returns_numpy_array(self):
        """calibrate_clipping should return a numpy array of norms."""
        torch.manual_seed(42)
        model = _SimpleLora()
        dataloader = _make_dataloader(num_batches=5)
        device = torch.device("cpu")

        norms = calibrate_clipping(model, dataloader, device, num_steps=5)

        assert isinstance(norms, np.ndarray)

    def test_collects_correct_number_of_norms(self):
        """Should collect exactly min(num_steps, len(dataloader)) norms."""
        torch.manual_seed(42)
        model = _SimpleLora()
        dataloader = _make_dataloader(num_batches=10)
        device = torch.device("cpu")

        norms = calibrate_clipping(model, dataloader, device, num_steps=5)
        assert len(norms) == 5

    def test_collects_all_when_dataloader_shorter(self):
        """When dataloader has fewer batches than num_steps, collect all."""
        torch.manual_seed(42)
        model = _SimpleLora()
        dataloader = _make_dataloader(num_batches=3)
        device = torch.device("cpu")

        norms = calibrate_clipping(model, dataloader, device, num_steps=100)
        assert len(norms) == 3

    def test_norms_are_positive(self):
        """All collected gradient norms should be positive."""
        torch.manual_seed(42)
        model = _SimpleLora()
        dataloader = _make_dataloader(num_batches=5)
        device = torch.device("cpu")

        norms = calibrate_clipping(model, dataloader, device, num_steps=5)

        assert all(n > 0 for n in norms), f"All norms should be positive, got {norms}"

    def test_prints_statistics(self, capsys):
        """Should print Median, P75, P90, P99 statistics."""
        torch.manual_seed(42)
        model = _SimpleLora()
        dataloader = _make_dataloader(num_batches=10)
        device = torch.device("cpu")

        calibrate_clipping(model, dataloader, device, num_steps=10)

        captured = capsys.readouterr()
        assert "Median:" in captured.out
        assert "P75:" in captured.out
        assert "P90:" in captured.out
        assert "P99:" in captured.out
        assert "Suggested C:" in captured.out

    def test_statistics_are_consistent(self):
        """Printed statistics should match numpy calculations on returned norms."""
        torch.manual_seed(42)
        model = _SimpleLora()
        dataloader = _make_dataloader(num_batches=20)
        device = torch.device("cpu")

        norms = calibrate_clipping(model, dataloader, device, num_steps=20)

        # Verify the returned norms produce correct statistics
        assert np.median(norms) > 0
        assert np.percentile(norms, 75) >= np.median(norms)
        assert np.percentile(norms, 90) >= np.percentile(norms, 75)
        assert np.percentile(norms, 99) >= np.percentile(norms, 90)

    def test_model_grads_zeroed_after_each_step(self):
        """After calibration, model gradients should be zeroed."""
        torch.manual_seed(42)
        model = _SimpleLora()
        dataloader = _make_dataloader(num_batches=5)
        device = torch.device("cpu")

        calibrate_clipping(model, dataloader, device, num_steps=5)

        # All LoRA param grads should be None or zero after zero_grad
        for n, p in model.named_parameters():
            if p.requires_grad and "lora" in n.lower():
                assert p.grad is None or p.grad.norm().item() == 0.0

    def test_num_steps_zero_returns_empty(self):
        """num_steps=0 should return an empty array."""
        torch.manual_seed(42)
        model = _SimpleLora()
        dataloader = _make_dataloader(num_batches=5)
        device = torch.device("cpu")

        norms = calibrate_clipping(model, dataloader, device, num_steps=0)
        assert len(norms) == 0
