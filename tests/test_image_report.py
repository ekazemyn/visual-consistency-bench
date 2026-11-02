"""Offline completeness, escaping, preview semantics, and bounded report output."""

import base64
import copy
import io
import json
from html.parser import HTMLParser
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from visual_consistency.cli import evaluate_manifest, main
from visual_consistency.images import LoadedSample
from visual_consistency.manifest import ValidationError, load_manifest
from visual_consistency.report import (
    MAX_PREVIEW_SIDE,
    MAX_REPORT_BYTES,
    _preview_panels,
    render_report,
)

DEMO = Path(__file__).resolve().parents[1] / "examples/images/manifest.json"


class Document(HTMLParser):
    def __init__(self, text):
        super().__init__()
        self.tags = []
        self.images = []
        self.text = []
        self.references = []
        self.feed(text)

    def handle_starttag(self, tag, attrs):
        self.tags.append(tag)
        attrs = dict(attrs)
        assert not any(name.startswith("on") for name in attrs)
        if tag == "img":
            self.images.append(attrs)
        for name in ("src", "href"):
            if name in attrs:
                self.references.append(attrs[name])

    def handle_data(self, data):
        self.text.append(data)


@pytest.fixture
def evaluated():
    manifest = load_manifest(DEMO)
    return manifest, evaluate_manifest(manifest)


def test_demo_is_self_contained_with_complete_previews(evaluated):
    manifest, result = evaluated
    report = render_report(result, manifest)
    document = Document(report)
    assert document.tags.count("article") == 8
    assert len(document.images) == 32
    assert "script" not in document.tags
    assert "link" not in document.tags
    assert "iframe" not in document.tags
    assert all(ref.startswith("data:image/png;base64,") for ref in document.references)
    for image in document.images:
        with Image.open(io.BytesIO(base64.b64decode(image["src"].split(",", 1)[1]))) as preview:
            preview.load()
            assert preview.format == "PNG"
            assert max(preview.size) <= MAX_PREVIEW_SIDE
            assert preview.size == (int(image["width"]), int(image["height"]))
            assert image["alt"]
    assert len(report.encode("utf-8")) <= MAX_REPORT_BYTES
    assert str(DEMO.parent) not in report
    assert "Perfect match (unbounded)" in report
    assert "not_applicable" in report
    assert "empty_preservation_region" in report
    assert "No preservation pixels are available" in report
    assert "Black = 0; white = 1" in report
    assert "7×7" in report
    assert "Content-Security-Policy" in report


def test_hostile_text_and_filenames_are_escaped(evaluated):
    manifest, result = copy.deepcopy(evaluated)
    attack = '</p><script>alert("x")</script><img src=x onerror="alert(1)">'
    result["samples"][0]["instruction"] = attack
    result["samples"][0]["provenance"]["source"] = attack
    result["samples"][0]["warnings"][0]["message"] = attack
    report = render_report(result, manifest)
    document = Document(report)
    assert "script" not in document.tags
    assert len(document.images) == 32
    assert attack in "".join(document.text)
    assert "&lt;script&gt;" in report
    assert attack not in report


def test_supplied_filename_is_escaped(tmp_path):
    data = json.loads(DEMO.read_text(encoding="utf-8"))
    data["samples"] = data["samples"][:1]
    record = data["samples"][0]
    for field in ("original_path", "edited_path", "mask_path"):
        name = "original & source.png" if field == "original_path" else Path(record[field]).name
        (tmp_path / name).write_bytes((DEMO.parent / record[field]).read_bytes())
        record[field] = name
    record["provenance"]["license_reference"] = None
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    manifest = load_manifest(path)
    report = render_report(evaluate_manifest(manifest), manifest)
    assert "original &amp; source.png" in report
    assert "original & source.png" in "".join(Document(report).text)


def test_failed_sample_and_not_applicable_remain_visible(evaluated):
    manifest, result = evaluated
    failed = result["samples"][0]
    failed.update(
        status="error",
        support=None,
        metrics={},
        errors=[{"code": "invalid_image", "message": "Original could not be decoded."}],
    )
    result["counts"].update(evaluated=7, failed=1)
    report = render_report(result, manifest)
    document = Document(report)
    assert document.tags.count("article") == 8
    assert len(document.images) == 28
    assert "No measurements: sample failed" in report
    assert "invalid_image" in report
    assert "no-preservation-support" in report
    assert "not_applicable" in report


def test_overlay_and_fixed_difference_scale():
    original = np.zeros((2, 2, 3), dtype=np.uint8)
    edited = original.copy()
    edited[0, 0] = [255, 0, 0]
    edited[1, 1] = 255
    mask = np.array([[True, False], [False, False]])
    loaded = LoadedSample({}, original, edited, mask, {}, [])
    panels = _preview_panels(loaded)
    overlay = np.asarray(panels[2])
    difference = np.asarray(panels[3])
    np.testing.assert_array_equal(overlay[0, 0], [102, 0, 102])
    np.testing.assert_array_equal(overlay[1, 1], [0, 0, 0])
    assert difference[0, 0] == 85
    assert difference[0, 1] == 0
    assert difference[1, 1] == 255
    np.testing.assert_array_equal(loaded.original, original)


def test_preview_budget_retains_all_metrics(evaluated, monkeypatch):
    monkeypatch.setattr("visual_consistency.report.MAX_PREVIEW_BYTES", 1)
    manifest, result = evaluated
    report = render_report(result, manifest)
    document = Document(report)
    assert not document.images
    assert document.tags.count("article") == 8
    assert document.tags.count("table") == 8
    assert "Preview budget reached" in report


def test_changed_or_missing_inputs_do_not_show_stale_previews(evaluated, monkeypatch):
    manifest, result = evaluated
    monkeypatch.setattr(
        "visual_consistency.report.hashlib.file_digest",
        lambda *args: type("Digest", (), {"hexdigest": lambda self: "changed"})(),
    )
    report = render_report(result, manifest)
    assert "input bytes changed" in report
    assert not Document(report).images

    def unavailable(*args):
        raise ValidationError("missing_asset", "Missing input.")

    monkeypatch.setattr("visual_consistency.report.load_sample", unavailable)
    report = render_report(result, manifest)
    assert "inputs could not be reloaded" in report
    assert Document(report).tags.count("table") == 8


def test_report_limit_retains_full_json_csv(tmp_path, monkeypatch):
    monkeypatch.setattr("visual_consistency.report.MAX_REPORT_BYTES", 2048)
    assert main([str(DEMO), str(tmp_path)]) == 2
    assert (
        json.loads((tmp_path / "results.json").read_text(encoding="utf-8"))["counts"]["evaluated"]
        == 8
    )
    assert (tmp_path / "metrics.csv").is_file()
    assert not (tmp_path / "report.html").exists()


def test_html_alone_is_protected_from_overwrite(tmp_path):
    report = tmp_path / "report.html"
    report.write_text("Existing report", encoding="utf-8")
    assert main([str(DEMO), str(tmp_path)]) == 2
    assert report.read_text(encoding="utf-8") == "Existing report"
    assert not (tmp_path / "results.json").exists()


def test_input_named_report_html_cannot_be_replaced(tmp_path):
    manifest = tmp_path / "report.html"
    manifest.write_bytes(DEMO.read_bytes())
    assert main([str(manifest), str(tmp_path), "--overwrite"]) == 2
    assert manifest.read_bytes() == DEMO.read_bytes()
