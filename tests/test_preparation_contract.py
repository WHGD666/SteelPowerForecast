from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "prepared_v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class PreparationContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.train = pd.read_csv(OUT / "train_prepared.csv", parse_dates=["datetime"])
        cls.test = pd.read_csv(OUT / "test_prepared.csv", parse_dates=["datetime"])
        cls.input = pd.read_csv(OUT / "input.csv", parse_dates=["datetime"])
        cls.manifest = json.loads((OUT / "preparation_manifest.json").read_text(encoding="utf-8"))

    def test_time_grid_and_schema(self) -> None:
        self.assertEqual(len(self.train), 11521)
        self.assertEqual(len(self.test), 192)
        self.assertEqual(self.train.columns.tolist(), self.test.columns.tolist())
        self.assertFalse(self.train["datetime"].duplicated().any())
        self.assertFalse(self.test["datetime"].duplicated().any())
        self.assertTrue((self.train["datetime"].diff().dropna() == pd.Timedelta(minutes=15)).all())
        self.assertTrue((self.test["datetime"].diff().dropna() == pd.Timedelta(minutes=15)).all())

    def test_input_has_no_missing_and_no_raw_targets(self) -> None:
        self.assertFalse(self.input.isna().any().any())
        self.assertNotIn("generator_1", self.input.columns)
        self.assertNotIn("generator_all", self.input.columns)
        self.assertIn("feat_current_generator_1", self.input.columns)
        self.assertIn("feat_current_generator_all", self.input.columns)
        self.assertNotIn("feat_boundary_context_overlap", self.input.columns)
        self.assertFalse(any(self.input[column].nunique(dropna=False) <= 1 for column in self.input.columns if column != "datetime"))
        self.assertTrue(all(column == "datetime" or not column.startswith("generator_") or column.startswith("generator_use_") for column in self.input.columns if not column.startswith("feat_")))

    def test_generated_features_use_required_prefix(self) -> None:
        raw_columns = set()
        for family in ("gas", "gas_holder", "gas_user", "load"):
            raw_columns.update(pd.read_csv(ROOT / "data" / f"Pre_{family}.csv", nrows=0).columns)
        extras = [column for column in self.input.columns if column not in raw_columns and column != "datetime"]
        self.assertTrue(extras)
        self.assertTrue(all(column.startswith("feat_") for column in extras))

    def test_target_labels_are_not_filled(self) -> None:
        self.assertEqual(int(self.train["generator_1"].notna().sum()), 11519)
        self.assertEqual(int(self.train["generator_all"].notna().sum()), 11519)
        self.assertEqual(int(self.test["generator_1"].notna().sum()), 190)
        self.assertEqual(int(self.test["generator_all"].notna().sum()), 191)

    def test_known_missing_value_is_causally_filled(self) -> None:
        indexed = self.train.set_index("datetime")
        missing_time = pd.Timestamp("2025-02-17 18:00:00")
        previous_time = missing_time - pd.Timedelta(minutes=15)
        self.assertEqual(indexed.loc[missing_time, "blast_furnace_1"], indexed.loc[previous_time, "blast_furnace_1"])
        self.assertEqual(indexed.loc[missing_time, "feat_missing_blast_furnace_1"], 1)

    def test_inserted_rows_and_boundary_are_flagged(self) -> None:
        indexed = self.train.set_index("datetime")
        for timestamp in (pd.Timestamp("2025-04-28 18:00:00"), pd.Timestamp("2025-04-28 18:15:00")):
            self.assertEqual(indexed.loc[timestamp, "feat_inserted_timestamp"], 1)
        boundary = pd.Timestamp("2025-05-01 00:00:00")
        self.assertEqual(indexed.loc[boundary, "feat_boundary_context_overlap"], 1)
        self.assertEqual(self.test.set_index("datetime").loc[boundary, "feat_boundary_context_overlap"], 1)

    def test_manifest_source_hashes_match_raw_files(self) -> None:
        for relative, expected in self.manifest["source_hashes"].items():
            self.assertEqual(sha256(ROOT / relative), expected)


if __name__ == "__main__":
    unittest.main()
