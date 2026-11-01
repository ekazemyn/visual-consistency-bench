"""Analytical pixel differences and spatial support checks for preservation metrics."""

import json
import math
from pathlib import Path

import numpy as np
import pytest

from visual_consistency.images import LoadedSample, load_sample
from visual_consistency.manifest import load_manifest
from visual_consistency.metrics import measure_preservation


def sample(size=9):
    original = np.zeros((size, size, 3), dtype=np.uint8)
    return LoadedSample({}, original, original.copy(), np.zeros((size, size), bool), {}, [])


def test_identity_and_strict_json():
    result = measure_preservation(sample())
    metrics = result["metrics"]
    assert metrics["preservation_mae"]["value"] == 0
    assert metrics["preservation_psnr"]["value"] is None
    assert metrics["preservation_psnr"]["status"] == "perfect_match"
    assert metrics["preservation_ssim"]["value"] == 1
    assert result["support"]["ssim_windows"] == 9
    assert metrics["inside_mask_mae"]["status"] == "not_applicable"
    json.dumps(result, allow_nan=False)


def test_known_single_channel_difference():
    loaded = sample(1)
    loaded.edited[0, 0, 0] = 255
    result = measure_preservation(loaded)
    assert result["metrics"]["preservation_mae"]["value"] == pytest.approx(1 / 3)
    assert result["metrics"]["preservation_psnr"]["value"] == pytest.approx(10 * math.log10(3))
    assert result["metrics"]["preservation_ssim"]["status"] == "not_applicable"


def test_subtraction_does_not_wrap_and_uses_byte_normalization():
    loaded = sample(7)
    loaded.original[:] = 255
    loaded.edited[:] = 254
    metrics = measure_preservation(loaded)["metrics"]
    assert metrics["preservation_mae"]["value"] == pytest.approx(1 / 255)
    assert metrics["preservation_psnr"]["value"] == pytest.approx(20 * math.log10(255))
    assert metrics["preservation_ssim"]["support_count"] == 1


def test_mask_excludes_every_window_touching_edit():
    loaded = sample(15)
    loaded.edit_mask[7, 7] = True
    loaded.edited[7, 7] = 255
    result = measure_preservation(loaded)
    assert result["support"]["ssim_windows"] == 81 - 49
    assert result["metrics"]["preservation_ssim"]["value"] == pytest.approx(1)
    assert result["metrics"]["preservation_mae"]["value"] == 0
    assert result["metrics"]["inside_mask_mae"]["value"] == 1
    assert result["metrics"]["inside_mask_mae"]["direction"] == "descriptive"


def test_ssim_constant_patch_has_analytical_value():
    loaded = sample(7)
    loaded.edited[:] = 255
    metrics = measure_preservation(loaded)["metrics"]
    assert metrics["preservation_mae"]["value"] == 1
    assert metrics["preservation_psnr"]["value"] == 0
    assert metrics["preservation_ssim"]["value"] == pytest.approx(0.0001 / 1.0001)


@pytest.mark.parametrize("size", [1, 6, 7, 9])
def test_empty_preservation_is_not_perfect(size):
    loaded = sample(size)
    loaded.edit_mask[:] = True
    result = measure_preservation(loaded)
    for name in ("preservation_mae", "preservation_psnr", "preservation_ssim"):
        assert result["metrics"][name]["status"] == "not_applicable"
        assert result["metrics"][name]["value"] is None
        assert result["metrics"][name]["support_count"] == 0
    json.dumps(result, allow_nan=False)


def test_fragmented_support_has_pixels_but_no_windows():
    loaded = sample()
    loaded.edit_mask[::2] = True
    result = measure_preservation(loaded)
    assert result["metrics"]["preservation_mae"]["status"] == "ok"
    assert result["metrics"]["preservation_ssim"]["status"] == "not_applicable"
    assert result["warnings"][0]["code"] == "insufficient_ssim_support"


def test_demo_metrics_are_finite_and_do_not_mutate_inputs():
    manifest = load_manifest(Path(__file__).resolve().parents[1] / "examples/images/manifest.json")
    for record in manifest.samples:
        loaded = load_sample(manifest, record)
        before = [array.copy() for array in (loaded.original, loaded.edited, loaded.edit_mask)]
        result = measure_preservation(loaded)
        json.dumps(result, allow_nan=False)
        for actual, expected in zip((loaded.original, loaded.edited, loaded.edit_mask), before):
            np.testing.assert_array_equal(actual, expected)
        assert (
            sum(result["support"][key] for key in ("preservation_pixels", "permitted_edit_pixels"))
            == 256 * 256
        )
