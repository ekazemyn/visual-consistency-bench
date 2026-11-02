"""End-to-end result contracts, failure retention, and output protection."""

import copy
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

from visual_consistency.cli import main

DEMO = Path(__file__).resolve().parents[1] / "examples/images/manifest.json"


def read_result(output):
    def reject(value):
        raise AssertionError(f"Nonstandard JSON: {value}")

    return json.loads((output / "results.json").read_text(encoding="utf-8"), parse_constant=reject)


def test_demo_repeatability_and_csv(tmp_path):
    first, second = tmp_path / "first", tmp_path / "second"
    assert main([str(DEMO), str(first)]) == 0
    assert main([str(DEMO), str(second)]) == 0
    result = read_result(first)
    assert result["counts"] == {"submitted": 8, "evaluated": 8, "failed": 0}
    assert result["manifest_sha256"] == hashlib.sha256(DEMO.read_bytes()).hexdigest()
    records = json.loads(DEMO.read_text(encoding="utf-8"))["samples"]
    for sample, record in zip(result["samples"], records):
        assert sample["sample_id"] == record["sample_id"]
        for field, digest in sample["input_sha256"].items():
            assert digest == hashlib.sha256((DEMO.parent / record[field]).read_bytes()).hexdigest()
        assert sample["width"] == sample["height"] == 256
        assert sample["provenance"] == record["provenance"]
    assert str(DEMO.parent) not in (first / "results.json").read_text(encoding="utf-8")
    assert "original_path" not in result["samples"][0]
    with (first / "metrics.csv").open(encoding="utf-8", newline="") as source:
        rows = list(csv.DictReader(source))
    assert len(rows) == 32
    assert {row["sample_id"] for row in rows} == {record["sample_id"] for record in records}
    assert any(row["metric_status"] == "perfect_match" and row["value"] == "" for row in rows)
    assert any(row["metric_status"] == "not_applicable" for row in rows)
    other = read_result(second)
    for run in (result, other):
        run.pop("started_at_utc")
        assert run.pop("wall_duration_seconds") >= 0
        for sample in run["samples"]:
            assert sample.pop("wall_duration_seconds") >= 0
    assert result == other


def test_overwrite_requires_option(tmp_path):
    output = tmp_path / "run"
    assert main([str(DEMO), str(output)]) == 0
    before = (output / "results.json").read_bytes()
    assert main([str(DEMO), str(output)]) == 2
    assert (output / "results.json").read_bytes() == before
    assert main([str(DEMO), str(output), "--overwrite"]) == 0


def test_bad_manifest_writes_no_success_output(tmp_path):
    manifest = tmp_path / "bad.json"
    manifest.write_text("{}", encoding="utf-8")
    output = tmp_path / "run"
    assert main([str(manifest), str(output)]) == 2
    assert not output.exists()


def test_failed_sample_retained_and_later_sample_evaluated(tmp_path):
    data = json.loads(DEMO.read_text(encoding="utf-8"))
    for record in data["samples"]:
        for field in ("original_path", "edited_path", "mask_path"):
            path = tmp_path / record[field]
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes((DEMO.parent / record[field]).read_bytes())
        record["provenance"]["license_reference"] = None
    data["samples"][0]["edited_path"] = "missing.png"
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(data), encoding="utf-8")
    output = tmp_path / "run"
    assert main([str(manifest), str(output)]) == 1
    result = read_result(output)
    assert result["counts"] == {"submitted": 8, "evaluated": 7, "failed": 1}
    failed = result["samples"][0]
    assert failed["sample_id"] == data["samples"][0]["sample_id"]
    assert failed["status"] == "error"
    assert failed["metrics"] == {}
    assert failed["width"] is None
    assert failed["input_sha256"]["edited_path"] is None
    assert failed["errors"][0]["code"] == "missing_asset"
    assert result["samples"][1]["status"] == "ok"
    with (output / "metrics.csv").open(encoding="utf-8", newline="") as source:
        rows = list(csv.DictReader(source))
    assert len(rows) == 29
    assert rows[0]["sample_status"] == "error"
    assert rows[0]["metric"] == ""


def test_processing_failure_is_not_fabricated_measurement(tmp_path, monkeypatch):
    def broken(sample):
        raise RuntimeError("private/path")

    monkeypatch.setattr("visual_consistency.cli.measure_preservation", broken)
    output = tmp_path / "run"
    assert main([str(DEMO), str(output)]) == 1
    result = read_result(output)
    assert result["counts"]["failed"] == 8
    assert all(not sample["metrics"] for sample in result["samples"])
    assert "private/path" not in json.dumps(result)


def test_nonfinite_measurement_fails_sample(tmp_path, monkeypatch):
    from visual_consistency.metrics import measure_preservation

    def broken(sample):
        result = copy.deepcopy(measure_preservation(sample))
        result["metrics"]["preservation_mae"]["value"] = float("nan")
        return result

    monkeypatch.setattr("visual_consistency.cli.measure_preservation", broken)
    output = tmp_path / "run"
    assert main([str(DEMO), str(output)]) == 1
    assert read_result(output)["counts"]["failed"] == 8


def test_output_cannot_replace_manifest_even_with_overwrite(tmp_path):
    manifest = tmp_path / "results.json"
    data = json.loads(DEMO.read_text(encoding="utf-8"))
    manifest.write_text(json.dumps(data), encoding="utf-8")
    before = manifest.read_bytes()
    assert main([str(manifest), str(tmp_path), "--overwrite"]) == 2
    assert manifest.read_bytes() == before


def test_module_command(tmp_path):
    completed = subprocess.run(
        [sys.executable, "-m", "visual_consistency", str(DEMO), str(tmp_path / "run")],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "8/8" in completed.stdout
