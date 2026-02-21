"""Unit tests for run metadata recording (Task 3.5).

Tests cover:
- append_metadata: JSONL file creation and appending
- _build_metadata: field completeness for completed and failed runs
- main() error handling: metadata recorded on training failure

Requirements: 7.1, 7.2, 7.3, 7.4
"""
import argparse
import json
import math
import os
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from train_dp_sft import append_metadata, _build_metadata


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_args(**overrides):
    """Create a minimal args namespace for _build_metadata."""
    defaults = {
        "seed": 42,
        "output_dir": "models/dp_sft_eps8_seed42",
        "training_data": "data/wiki_trimmed_with_canary_50.jsonl",
        "base_model": "models/Qwen2.5-0.5B-Instruct",
        "delta": 0.0001,
        "clipping_norm": 1.0,
        "accountant_type": "rdp",
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


# Required DP-specific fields (Requirement 7.2)
DP_FIELDS = {
    "target_epsilon", "actual_epsilon", "delta", "N_for_delta",
    "clipping_norm", "accountant_type", "dp_method", "noise_multiplier",
}

# Required standard fields (Requirement 7.3)
STANDARD_FIELDS = {
    "seed", "model_path", "training_data", "base_model",
    "timestamp", "hyperparams",
}


# ---------------------------------------------------------------------------
# Tests: append_metadata
# ---------------------------------------------------------------------------

class TestAppendMetadata:
    """Tests for append_metadata() JSONL writing."""

    def test_creates_directory_and_file(self, tmp_path):
        """append_metadata creates parent dirs if they don't exist."""
        filepath = str(tmp_path / "subdir" / "run_metadata.jsonl")
        append_metadata({"key": "value"}, filepath=filepath)

        assert os.path.isfile(filepath)
        with open(filepath, "r") as f:
            line = f.readline().strip()
        assert json.loads(line) == {"key": "value"}

    def test_appends_multiple_lines(self, tmp_path):
        """Each call appends one JSON line."""
        filepath = str(tmp_path / "run_metadata.jsonl")
        append_metadata({"run": 1}, filepath=filepath)
        append_metadata({"run": 2}, filepath=filepath)

        with open(filepath, "r") as f:
            lines = [json.loads(line) for line in f if line.strip()]
        assert len(lines) == 2
        assert lines[0]["run"] == 1
        assert lines[1]["run"] == 2

    def test_each_line_is_valid_json(self, tmp_path):
        """Each line in the JSONL file is independently parseable."""
        filepath = str(tmp_path / "run_metadata.jsonl")
        for i in range(5):
            append_metadata({"index": i, "nested": {"a": i * 2}},
                            filepath=filepath)

        with open(filepath, "r") as f:
            for line in f:
                if line.strip():
                    parsed = json.loads(line)
                    assert "index" in parsed


# ---------------------------------------------------------------------------
# Tests: _build_metadata
# ---------------------------------------------------------------------------

class TestBuildMetadata:
    """Tests for _build_metadata() field completeness."""

    def test_completed_run_has_all_dp_fields(self):
        """Requirement 7.2: DP-specific fields present."""
        args = _make_args()
        meta = _build_metadata(
            args,
            dp_method="opacus",
            target_epsilon=8.0,
            actual_epsilon=7.93,
            noise_multiplier=0.85,
            n_for_delta=10050,
        )
        for field in DP_FIELDS:
            assert field in meta, f"Missing DP field: {field}"

    def test_completed_run_has_all_standard_fields(self):
        """Requirement 7.3: standard fields present."""
        args = _make_args()
        meta = _build_metadata(
            args,
            dp_method="opacus",
            target_epsilon=8.0,
            actual_epsilon=7.93,
            noise_multiplier=0.85,
            n_for_delta=10050,
        )
        for field in STANDARD_FIELDS:
            assert field in meta, f"Missing standard field: {field}"
        assert meta["type"] == "dp_sft_training"
        assert meta["status"] == "completed"

    def test_hyperparams_contains_expected_keys(self):
        """Hyperparams dict has all training config keys."""
        args = _make_args()
        meta = _build_metadata(
            args,
            dp_method="opacus",
            target_epsilon=8.0,
            actual_epsilon=7.93,
            noise_multiplier=0.85,
            n_for_delta=10050,
        )
        hp = meta["hyperparams"]
        expected_keys = {
            "learning_rate", "num_train_epochs",
            "per_device_train_batch_size", "gradient_accumulation_steps",
            "lora_r", "lora_alpha", "lora_dropout",
        }
        assert set(hp.keys()) == expected_keys

    def test_failed_run_has_error_field(self):
        """Requirement 7.4: failed status includes error message."""
        args = _make_args()
        meta = _build_metadata(
            args,
            dp_method="opacus",
            target_epsilon=8.0,
            actual_epsilon=0.0,
            noise_multiplier=0.0,
            n_for_delta=10050,
            status="failed",
            error="RuntimeError: CUDA out of memory",
        )
        assert meta["status"] == "failed"
        assert meta["error"] == "RuntimeError: CUDA out of memory"

    def test_completed_run_has_no_error_field(self):
        """Completed runs should not have an error field."""
        args = _make_args()
        meta = _build_metadata(
            args,
            dp_method="opacus",
            target_epsilon=8.0,
            actual_epsilon=7.93,
            noise_multiplier=0.85,
            n_for_delta=10050,
        )
        assert "error" not in meta

    def test_infinity_epsilon_serialized_as_string(self):
        """JSON doesn't support infinity; use string 'inf'."""
        args = _make_args()
        meta = _build_metadata(
            args,
            dp_method="none",
            target_epsilon=math.inf,
            actual_epsilon=math.inf,
            noise_multiplier=0.0,
            n_for_delta=10050,
        )
        assert meta["target_epsilon"] == "inf"
        assert meta["actual_epsilon"] == "inf"
        # Verify it's JSON-serializable
        json.dumps(meta)

    def test_standard_sft_metadata_values(self):
        """Standard SFT (epsilon=inf) uses dp_method='none'."""
        args = _make_args()
        meta = _build_metadata(
            args,
            dp_method="none",
            target_epsilon="inf",
            actual_epsilon="inf",
            noise_multiplier=0.0,
            n_for_delta=10050,
        )
        assert meta["dp_method"] == "none"
        assert meta["noise_multiplier"] == 0.0

    def test_manual_dp_metadata_values(self):
        """Manual DP fallback uses dp_method='manual_lora_dp'."""
        args = _make_args()
        meta = _build_metadata(
            args,
            dp_method="manual_lora_dp",
            target_epsilon=4.0,
            actual_epsilon=3.95,
            noise_multiplier=1.2,
            n_for_delta=10050,
        )
        assert meta["dp_method"] == "manual_lora_dp"
        assert meta["actual_epsilon"] == 3.95

    def test_timestamp_format(self):
        """Timestamp is UTC ISO format (ends with +00:00 or Z)."""
        args = _make_args()
        meta = _build_metadata(
            args,
            dp_method="opacus",
            target_epsilon=8.0,
            actual_epsilon=7.93,
            noise_multiplier=0.85,
            n_for_delta=10050,
        )
        ts = meta["timestamp"]
        assert ts.endswith("+00:00") or ts.endswith("Z")
        assert "T" in ts

    def test_n_for_delta_recorded(self):
        """N_for_delta reflects the training sample count."""
        args = _make_args()
        meta = _build_metadata(
            args,
            dp_method="opacus",
            target_epsilon=8.0,
            actual_epsilon=7.93,
            noise_multiplier=0.85,
            n_for_delta=9999,
        )
        assert meta["N_for_delta"] == 9999
