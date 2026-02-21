"""Tests for src/run_dp_sft_audit.py

Covers: parse_args, resolve_model_dirs, extract_run_info,
        run_audit_for_model, write_results_csv, and CSV output format.
Requirements: 4.1, 4.2, 4.3, 4.4
"""

import os

import pandas as pd
import pytest

from src.run_dp_sft_audit import (
    CSV_COLUMNS,
    PRIMARY_METRICS,
    SECONDARY_METRICS,
    _stub_metrics,
    extract_run_info,
    parse_args,
    resolve_model_dirs,
    run_audit_for_model,
    write_results_csv,
)


# ── parse_args ──────────────────────────────────────────────────────

class TestParseArgs:
    def test_required_model_dirs(self):
        with pytest.raises(SystemExit):
            parse_args([])

    def test_defaults(self):
        args = parse_args(["--model-dirs", "models/dp_sft_eps_inf_seed42"])
        assert args.model_dirs == "models/dp_sft_eps_inf_seed42"
        assert args.output_csv == "reports/dp_sft_audit_results.csv"
        assert args.pilot is False
        assert args.training_data == "data/wiki_trimmed_with_canary_50.jsonl"
        assert args.base_model == "models/Qwen2.5-0.5B-Instruct"
        assert args.canary_file == "data/canary_output_50.txt"
        assert args.allow_stub is False

    def test_custom_values(self):
        args = parse_args([
            "--model-dirs", "a,b",
            "--output-csv", "out.csv",
            "--pilot",
            "--training-data", "data.jsonl",
            "--base-model", "m/base",
        ])
        assert args.model_dirs == "a,b"
        assert args.output_csv == "out.csv"
        assert args.pilot is True
        assert args.training_data == "data.jsonl"
        assert args.base_model == "m/base"


# ── resolve_model_dirs ──────────────────────────────────────────────

class TestResolveModelDirs:
    def test_comma_separated(self, tmp_path):
        d1 = tmp_path / "dp_sft_eps_inf_seed42"
        d2 = tmp_path / "dp_sft_eps8_seed42"
        d1.mkdir()
        d2.mkdir()
        result = resolve_model_dirs(f"{d1},{d2}")
        assert len(result) == 2
        assert str(d1) in result or os.path.normpath(str(d1)) in result

    def test_glob_pattern(self, tmp_path):
        for name in ["dp_sft_eps_inf_seed42", "dp_sft_eps8_seed42", "dp_sft_eps4_seed42"]:
            (tmp_path / name).mkdir()
        pattern = str(tmp_path / "dp_sft_eps*")
        result = resolve_model_dirs(pattern)
        assert len(result) == 3

    def test_nonexistent_dirs_filtered(self, tmp_path):
        existing = tmp_path / "dp_sft_eps_inf_seed42"
        existing.mkdir()
        result = resolve_model_dirs(f"{existing},{tmp_path / 'nope'}")
        assert len(result) == 1

    def test_empty_string(self):
        result = resolve_model_dirs("")
        assert result == []

    def test_deduplication(self, tmp_path):
        d = tmp_path / "dp_sft_eps_inf_seed42"
        d.mkdir()
        result = resolve_model_dirs(f"{d},{d}")
        assert len(result) == 1

    def test_sorted_output(self, tmp_path):
        names = ["dp_sft_eps8_seed42", "dp_sft_eps4_seed42", "dp_sft_eps_inf_seed42"]
        for n in names:
            (tmp_path / n).mkdir()
        result = resolve_model_dirs(str(tmp_path / "dp_sft_eps*"))
        assert result == sorted(result)

    def test_files_excluded(self, tmp_path):
        (tmp_path / "dp_sft_eps_inf_seed42").touch()  # file, not dir
        result = resolve_model_dirs(str(tmp_path / "dp_sft_eps*"))
        assert result == []


# ── extract_run_info ────────────────────────────────────────────────

class TestExtractRunInfo:
    def test_eps_inf(self):
        info = extract_run_info("models/dp_sft_eps_inf_seed42")
        assert info["epsilon"] == "inf"
        assert info["seed"] == 42

    def test_eps_numeric(self):
        info = extract_run_info("models/dp_sft_eps8_seed123")
        assert info["epsilon"] == "8"
        assert info["seed"] == 123

    def test_eps_float(self):
        info = extract_run_info("models/dp_sft_eps0.5_seed456")
        assert info["epsilon"] == "0.5"
        assert info["seed"] == 456

    def test_model_dir_preserved(self):
        path = "/some/path/dp_sft_eps4_seed42"
        info = extract_run_info(path)
        assert info["model_dir"] == path

    def test_invalid_dir_raises(self):
        with pytest.raises(ValueError, match="Cannot parse"):
            extract_run_info("models/some_random_dir")

    def test_trailing_slash(self):
        info = extract_run_info("models/dp_sft_eps1_seed42/")
        assert info["epsilon"] == "1"
        assert info["seed"] == 42


# ── run_audit_for_model ─────────────────────────────────────────────

class TestRunAuditForModel:
    def test_non_pilot_returns_all_numeric(self):
        result = run_audit_for_model("dir", "data", "base", pilot=False, allow_stub=True)
        for m in PRIMARY_METRICS + SECONDARY_METRICS:
            assert m in result
            assert isinstance(result[m], (int, float))

    def test_pilot_primary_numeric(self):
        result = run_audit_for_model("dir", "data", "base", pilot=True, allow_stub=True)
        for m in PRIMARY_METRICS:
            assert isinstance(result[m], (int, float))

    def test_pilot_secondary_na(self):
        result = run_audit_for_model("dir", "data", "base", pilot=True, allow_stub=True)
        for m in SECONDARY_METRICS:
            assert result[m] == "NA"

    def test_stub_has_audit_status(self):
        result = run_audit_for_model("dir", "data", "base", allow_stub=True)
        assert result["audit_status"] == "stub"
        assert result["fallback_reason"] != ""

    def test_fail_fast_without_allow_stub(self):
        with pytest.raises(RuntimeError, match="--allow-stub"):
            run_audit_for_model("dir", "data", "base", allow_stub=False)


# ── write_results_csv ───────────────────────────────────────────────

class TestWriteResultsCsv:
    def _make_rows(self, pilot=False):
        rows = []
        for eps, seed in [("inf", 42), ("8", 42)]:
            row = {"epsilon": eps, "seed": seed}
            for m in PRIMARY_METRICS:
                row[m] = -6.0
            for m in SECONDARY_METRICS:
                row[m] = "NA" if pilot else 0.5
            row["audit_status"] = "real"
            row["fallback_reason"] = ""
            rows.append(row)
        return rows

    def test_csv_columns_match(self, tmp_path):
        csv_path = str(tmp_path / "out.csv")
        write_results_csv(self._make_rows(), csv_path)
        df = pd.read_csv(csv_path)
        assert list(df.columns) == CSV_COLUMNS

    def test_csv_row_count(self, tmp_path):
        csv_path = str(tmp_path / "out.csv")
        rows = self._make_rows()
        write_results_csv(rows, csv_path)
        df = pd.read_csv(csv_path)
        assert len(df) == len(rows)

    def test_pilot_na_in_csv(self, tmp_path):
        csv_path = str(tmp_path / "out.csv")
        write_results_csv(self._make_rows(pilot=True), csv_path)
        df = pd.read_csv(csv_path, keep_default_na=False)
        for m in SECONDARY_METRICS:
            assert (df[m] == "NA").all()

    def test_non_pilot_numeric_in_csv(self, tmp_path):
        csv_path = str(tmp_path / "out.csv")
        write_results_csv(self._make_rows(pilot=False), csv_path)
        df = pd.read_csv(csv_path)
        for m in SECONDARY_METRICS:
            assert pd.api.types.is_numeric_dtype(df[m])

    def test_creates_parent_dirs(self, tmp_path):
        csv_path = str(tmp_path / "sub" / "dir" / "out.csv")
        write_results_csv(self._make_rows(), csv_path)
        assert os.path.isfile(csv_path)

    def test_epsilon_column_values(self, tmp_path):
        csv_path = str(tmp_path / "out.csv")
        write_results_csv(self._make_rows(), csv_path)
        df = pd.read_csv(csv_path, dtype={"epsilon": str})
        assert set(df["epsilon"]) == {"inf", "8"}


# ── _stub_metrics ───────────────────────────────────────────────────

class TestStubMetrics:
    def test_non_pilot_all_numeric(self):
        result = _stub_metrics(pilot=False)
        for m in PRIMARY_METRICS + SECONDARY_METRICS:
            assert isinstance(result[m], (int, float))

    def test_pilot_secondary_na(self):
        result = _stub_metrics(pilot=True)
        for m in PRIMARY_METRICS:
            assert isinstance(result[m], (int, float))
        for m in SECONDARY_METRICS:
            assert result[m] == "NA"


# ── _load_canaries ──────────────────────────────────────────────────

class TestLoadCanaries:
    def test_strips_prefix(self, tmp_path):
        from src.run_dp_sft_audit import _load_canaries
        f = tmp_path / "canaries.txt"
        f.write_text("Canary 1: hello world\nCanary 2: foo bar\n")
        result = _load_canaries(str(f))
        assert result == ["hello world", "foo bar"]

    def test_skips_blank_lines(self, tmp_path):
        from src.run_dp_sft_audit import _load_canaries
        f = tmp_path / "canaries.txt"
        f.write_text("Canary 1: a\n\n\nCanary 2: b\n")
        result = _load_canaries(str(f))
        assert len(result) == 2

    def test_no_prefix(self, tmp_path):
        from src.run_dp_sft_audit import _load_canaries
        f = tmp_path / "canaries.txt"
        f.write_text("plain text line\n")
        result = _load_canaries(str(f))
        assert result == ["plain text line"]
