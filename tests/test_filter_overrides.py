"""Tests for filter_race_results with manual JSON overrides."""

import json
import pytest
import pandas as pd
from pathlib import Path
from unittest.mock import patch

from fellpace.filter import filter_race_results
from fellpace.race_overrides import build_override_key, set_race_override, _load_overrides


def _make_results(**kwargs):
    """Build a minimal racer_results DataFrame for testing."""
    defaults = {
        "Racer_ID": [1, 1, 1],
        "Racer_Name": ["alice", "alice", "alice"],
        "Race_Name": ["Grindleford", "Bamford", "Exterminator"],
        "Season": [2022, 2022, 2022],
        "Zpred_mu": [0.1, 0.2, 0.3],
        "Zpred_sig": [0.05, 0.05, 0.05],
    }
    defaults.update(kwargs)
    return pd.DataFrame(defaults)


# ---------------------------------------------------------------------------
# build_override_key
# ---------------------------------------------------------------------------

class TestBuildOverrideKey:
    def test_uses_racer_id_when_present(self):
        row = {"Racer_ID": 42, "Racer_Name": "alice", "Season": 2023, "Race_Name": "Grindleford"}
        key = build_override_key(row)
        assert key == "42|2023|grindleford"

    def test_falls_back_to_racer_name(self):
        row = {"Racer_ID": None, "Racer_Name": "Alice Smith", "Season": 2023, "Race_Name": "Bamford"}
        key = build_override_key(row)
        assert key == "alice smith|2023|bamford"

    def test_normalises_race_name_case_and_whitespace(self):
        row = {"Racer_ID": 1, "Season": 2022, "Race_Name": "  Grindleford Fell Race  "}
        key = build_override_key(row)
        assert key == "1|2022|grindleford fell race"

    def test_returns_none_when_race_name_missing(self):
        row = {"Racer_ID": 1, "Racer_Name": "alice", "Season": 2022, "Race_Name": None}
        assert build_override_key(row) is None

    def test_returns_none_when_season_missing(self):
        row = {"Racer_ID": 1, "Racer_Name": "alice", "Season": None, "Race_Name": "Grindleford"}
        assert build_override_key(row) is None

    def test_returns_none_when_identity_missing(self):
        row = {"Racer_ID": None, "Racer_Name": None, "Season": 2022, "Race_Name": "Grindleford"}
        assert build_override_key(row) is None

    def test_works_with_pandas_series(self):
        row = pd.Series({"Racer_ID": 7, "Racer_Name": "bob", "Season": 2021, "Race_Name": "HopeWakes"})
        key = build_override_key(row)
        assert key == "7|2021|hopewakes"


# ---------------------------------------------------------------------------
# filter_race_results — no overrides (existing behaviour preserved)
# ---------------------------------------------------------------------------

class TestFilterNoOverrides:
    def test_excludes_list_race(self):
        df = _make_results()
        with patch("fellpace.filter._load_overrides", return_value={}):
            filter_race_results(df)
        exterminator_row = df[df["Race_Name"] == "Exterminator"]
        assert not exterminator_row["include"].all()

    def test_includes_normal_races(self):
        df = _make_results()
        with patch("fellpace.filter._load_overrides", return_value={}):
            filter_race_results(df)
        for race in ["Grindleford", "Bamford"]:
            assert df.loc[df["Race_Name"] == race, "include"].all()


# ---------------------------------------------------------------------------
# filter_race_results — with overrides
# ---------------------------------------------------------------------------

class TestFilterWithOverrides:
    def test_json_true_forces_inclusion_of_excluded_race(self):
        df = _make_results()
        # Exterminator is in EXCLUDE_LIST → normally excluded
        overrides = {"1|2022|exterminator": True}
        with patch("fellpace.filter._load_overrides", return_value=overrides):
            filter_race_results(df)
        assert df.loc[df["Race_Name"] == "Exterminator", "include"].all()

    def test_json_false_forces_exclusion_of_included_race(self):
        df = _make_results()
        overrides = {"1|2022|grindleford": False}
        with patch("fellpace.filter._load_overrides", return_value=overrides):
            filter_race_results(df)
        assert not df.loc[df["Race_Name"] == "Grindleford", "include"].all()

    def test_unrelated_key_does_not_affect_result(self):
        df = _make_results()
        overrides = {"999|2022|grindleford": False}  # different racer
        with patch("fellpace.filter._load_overrides", return_value=overrides):
            filter_race_results(df)
        assert df.loc[df["Race_Name"] == "Grindleford", "include"].all()

    def test_override_does_not_affect_rows_without_key_fields(self):
        """Synthetic rows missing Racer_ID and Racer_Name are silently skipped."""
        df = _make_results(
            Racer_ID=[None, 1, 1],
            Racer_Name=[None, "alice", "alice"],
        )
        overrides = {"1|2022|bamford": False}
        with patch("fellpace.filter._load_overrides", return_value=overrides):
            filter_race_results(df)
        # Row 0 has no identity — should not raise and should default to auto value
        assert "include" in df.columns


# ---------------------------------------------------------------------------
# set_race_override round-trip (uses tmp_path)
# ---------------------------------------------------------------------------

class TestSetRaceOverride:
    def test_round_trip(self, tmp_path):
        override_path = tmp_path / "overrides.json"
        with patch("fellpace.race_overrides.RACE_OVERRIDE_PATH", override_path):
            set_race_override(42, 2023, "Grindleford", True)
            loaded = _load_overrides(override_path)
        assert loaded["42|2023|grindleford"] is True

    def test_overwrites_existing_entry(self, tmp_path):
        override_path = tmp_path / "overrides.json"
        with patch("fellpace.race_overrides.RACE_OVERRIDE_PATH", override_path):
            set_race_override(42, 2023, "Grindleford", True)
            set_race_override(42, 2023, "Grindleford", False)
            loaded = _load_overrides(override_path)
        assert loaded["42|2023|grindleford"] is False

    def test_uses_name_fallback(self, tmp_path):
        override_path = tmp_path / "overrides.json"
        with patch("fellpace.race_overrides.RACE_OVERRIDE_PATH", override_path):
            set_race_override("Alice Smith", 2022, "Bamford", False)
            loaded = _load_overrides(override_path)
        assert "alice smith|2022|bamford" in loaded
