"""Tests for the config loader, cleaning log and provenance log.

Infrastructure, not methodology - but the config test doubles as a guard on
the methodological constants. If someone changes the SICR threshold or drops
the platform-risk exclusion, these fail.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.data.cleaning_log import CleaningLog
from src.data.provenance import (
    log_retrieval,
    read_provenance_log,
    sha256_checksum,
    verify_file,
)
from src.utils.config import (
    load_config,
    load_leakage_register,
    repo_root,
    resolve_path,
    set_global_seed,
)


class TestConfig:
    def test_loads_and_validates(self):
        config = load_config()
        assert config["seed"] == 20260815
        assert config["meta"]["student_number"] == "ST10517781"

    def test_partition_shares_sum_to_one(self):
        """Retained to reproduce the ICE Task 2 figures; not the study's scheme."""
        p = load_config()["partition"]["legacy_out_of_time"]
        assert p["train_share"] + p["validation_share"] + p["test_share"] == pytest.approx(1.0)
    
    def test_rolling_origin_folds_are_configured(self):
        """Three folds and the two embargo settings Section 3.4.3 reports."""
        rolling = load_config()["partition"]["rolling_origin"]
        assert len(rolling["folds"]) == 3
        assert rolling["embargo_months"] == 0
        assert rolling["embargo_months_robustness"] == 36

    def test_primary_term_scope_is_thirty_six_months(self):
        """60-month loans default at roughly twice the rate and, because of the
        maturity filter, exist only in cohorts to 2013, so term mix would be
        confounded with calendar time."""
        scope = load_config()["sample"]["term_scope"]
        assert scope["primary_term_months"] == 36
        assert scope["robustness_term_months"] == 60

    def test_methodological_constants(self):
        """Guards the numbers the methodology commits to in writing."""
        c = load_config()
        assert c["ecl"]["lgd"] == 0.45
        assert c["ecl"]["lgd_sensitivity"] == [0.30, 0.45, 0.60]
        assert c["staging"]["sicr"]["relative_threshold"] == 2.0
        assert c["staging"]["stage3_days_past_due"] == 90
        assert c["leakage"]["univariate_auc_flag_threshold"] == 0.90
        assert c["models"]["tuning_budget_trials"] == 60

    def test_model_registry_includes_post_lessmann_families(self):
        """Lessmann et al. (2015) predates LightGBM and CatBoost. A 2026
        benchmark that stops at XGBoost reads as dated."""
        families = load_config()["models"]["families"]
        assert "lightgbm" in families
        assert "catboost" in families

    def test_three_feature_blocks_present(self):
        """The A/B/C design is what separates 'macro conditioning in general'
        from 'South African macro specifically'."""
        blocks = load_config()["feature_blocks"]
        assert set(blocks) == {"A", "B", "C"}
        assert blocks["A"]["macro_panel"] is None
        assert blocks["B"]["macro_panel"] == "us"
        assert blocks["C"]["macro_panel"] == "za"

    def test_six_series_in_each_macro_panel(self):
        macro = load_config()["macro"]
        assert len(macro["za_series"]) == 6
        assert len(macro["us_series"]) == 6

    def test_publication_lags_are_specified(self):
        """Every series needs an explicit lag - this is the look-ahead control."""
        for panel in ("za_series", "us_series"):
            for series in load_config()["macro"][panel]:
                assert "publication_lag_months" in series, series["name"]
                assert series["publication_lag_months"] >= 0

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError, match="config not found"):
            load_config("does/not/exist.yaml")

    def test_bad_partition_shares_rejected(self, tmp_path):
        bad = tmp_path / "bad.yaml"
        bad.write_text(
            "meta: {}\nseed: 1\npaths: {}\nsample: {}\nleakage: {}\n"
            "partition: {legacy_out_of_time: "
            "{train_share: 0.5, validation_share: 0.3, test_share: 0.3}}\n"
        )
        with pytest.raises(ValueError, match="must sum to 1.0"):
            load_config(bad)

    def test_missing_section_rejected(self, tmp_path):
        bad = tmp_path / "bad.yaml"
        bad.write_text("seed: 1\n")
        with pytest.raises(ValueError, match="missing required sections"):
            load_config(bad)

    def test_resolve_path_is_absolute(self):
        assert resolve_path("data/raw").is_absolute()
        assert resolve_path("data/raw").parent.parent == repo_root()

    def test_set_global_seed_is_reproducible(self):
        import numpy as np

        set_global_seed(42)
        first = np.random.rand(5)
        set_global_seed(42)
        assert np.allclose(first, np.random.rand(5))


class TestLeakageRegister:
    def test_platform_risk_fields_excluded(self):
        """grade, sub_grade and int_rate encode Lending Club's own risk model.
        This is the exclusion most published work on this dataset omits."""
        register = load_leakage_register()
        platform = register["exclusions"]["platform_risk_assessment"]["fields"]
        assert set(platform) == {"grade", "sub_grade", "int_rate"}

    def test_hardship_pattern_present(self):
        register = load_leakage_register()
        patterns = register["exclusions"]["distress_arrangements"]["field_patterns"]
        assert "hardship_*" in patterns

    def test_int_rate_marked_operational(self):
        """Retained for ECL discounting, never a model input."""
        register = load_leakage_register()
        entry = next(
            e for e in register["retained_with_justification"] if e["field"] == "int_rate"
        )
        assert entry["excluded_from_modelling"] is True

    def test_empty_register_rejected(self, tmp_path):
        """An empty register would silently admit every leaked field, which is
        worse than having no register at all."""
        bad = tmp_path / "empty.yaml"
        bad.write_text("exclusions: {}\n")
        with pytest.raises(ValueError, match="no 'exclusions' section"):
            load_leakage_register(bad)


class TestCleaningLog:
    def test_records_counts_hand_calculated(self):
        """100 rows in, 60 out: 40 removed, exactly 40%. Two columns to one."""
        log = CleaningLog()
        before = pd.DataFrame({"a": range(100), "b": range(100)})
        after = before.head(60)[["a"]]
        log.record("drop", "Drop rows and a column", before, after)

        step = log.steps[0]
        assert step.rows_removed == 40
        assert step.rows_removed_pct == pytest.approx(0.40)
        assert step.cols_removed == 1

    def test_record_returns_frame_for_chaining(self):
        """Returning `after` makes it hard to filter and forget to log."""
        log = CleaningLog()
        before = pd.DataFrame({"a": range(10)})
        after = before.head(5)
        assert log.record("s", "d", before, after) is after

    def test_details_preserved(self):
        log = CleaningLog()
        frame = pd.DataFrame({"a": range(10)})
        log.record("maturity", "filter", frame, frame, default_rate_before=0.16)
        assert log.steps[0].details["default_rate_before"] == 0.16

    def test_summary_and_frame_render(self):
        log = CleaningLog()
        frame = pd.DataFrame({"a": range(10)})
        log.record("one", "first", frame, frame.head(8))
        log.record("two", "second", frame.head(8), frame.head(6))
        assert "NET" in log.summary()
        assert len(log.to_frame()) == 2

    def test_empty_log(self):
        assert "empty" in CleaningLog().summary().lower()

    def test_write_jsonl(self, tmp_path):
        log = CleaningLog()
        frame = pd.DataFrame({"a": range(10)})
        log.record("one", "first", frame, frame.head(8))
        path = tmp_path / "logs" / "cleaning.jsonl"
        log.write(path)
        assert len(path.read_text().strip().splitlines()) == 1


class TestProvenance:
    def test_known_checksum_hand_calculated(self, tmp_path):
        """SHA-256 of the empty string is a published constant."""
        empty = tmp_path / "empty.txt"
        empty.write_bytes(b"")
        assert sha256_checksum(empty) == (
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        )

    def test_abc_checksum_hand_calculated(self, tmp_path):
        """SHA-256("abc") is the canonical NIST test vector."""
        f = tmp_path / "abc.txt"
        f.write_bytes(b"abc")
        assert sha256_checksum(f) == (
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        )

    def test_streaming_matches_whole_file(self, tmp_path):
        """Chunked reading must give the same digest as reading it whole."""
        import hashlib

        big = tmp_path / "big.bin"
        payload = b"x" * (3 * (1 << 20) + 17)  # spans chunk boundaries
        big.write_bytes(payload)
        assert sha256_checksum(big) == hashlib.sha256(payload).hexdigest()

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="no such file"):
            sha256_checksum(tmp_path / "nope.csv")

    def test_log_and_read_roundtrip(self, tmp_path):
        data = tmp_path / "data.csv"
        data.write_text("a,b\n1,2\n")
        log = tmp_path / "provenance.jsonl"

        record = log_retrieval(data, "https://example.org/data", "CC-BY-4.0", log)
        assert record.filename == "data.csv"
        assert record.licence == "CC-BY-4.0"

        records = read_provenance_log(log)
        assert len(records) == 1
        assert records[0].sha256 == record.sha256

    def test_log_is_append_only(self, tmp_path):
        """Provenance records must never be overwritten - that is the point."""
        data = tmp_path / "data.csv"
        data.write_text("x")
        log = tmp_path / "provenance.jsonl"
        log_retrieval(data, "u", "l", log)
        log_retrieval(data, "u", "l", log)
        assert len(read_provenance_log(log)) == 2

    def test_verify_detects_modification(self, tmp_path):
        """The check that catches a raw file changing between pipeline steps."""
        data = tmp_path / "data.csv"
        data.write_text("original")
        digest = sha256_checksum(data)
        assert verify_file(data, digest)

        data.write_text("tampered")
        assert not verify_file(data, digest)

    def test_read_absent_log_returns_empty(self, tmp_path):
        assert read_provenance_log(tmp_path / "nothing.jsonl") == []
