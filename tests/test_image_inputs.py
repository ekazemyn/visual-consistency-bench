"""Contract-focused input checks using tiny local fixtures."""

import copy
import json
import struct
import zlib
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, PngImagePlugin

from visual_consistency.images import load_sample
from visual_consistency.manifest import ValidationError, load_manifest

DEMO = Path(__file__).resolve().parents[1] / "examples/images/manifest.json"


@pytest.fixture
def inputs(tmp_path):
    record = copy.deepcopy(json.loads(DEMO.read_text(encoding="utf-8"))["samples"][0])
    record.update(original_path="original.png", edited_path="edited.png", mask_path="mask.png")
    record["provenance"].update(
        source_url="https://example.org/controls", license_reference="LICENSE.txt"
    )
    (tmp_path / "LICENSE.txt").write_text("Fixture permission", encoding="utf-8")
    Image.new("RGB", (8, 8), (10, 20, 30)).save(tmp_path / "original.png")
    Image.new("RGB", (8, 8), (10, 20, 30)).save(tmp_path / "edited.png")
    Image.new("L", (8, 8), 0).save(tmp_path / "mask.png")
    path = tmp_path / "manifest.json"
    data = {"schema_version": "1.0", "dataset_id": "test", "samples": [record]}
    path.write_text(json.dumps(data), encoding="utf-8")
    return path, data


def write_manifest(inputs):
    path, data = inputs
    path.write_text(json.dumps(data), encoding="utf-8")
    return load_manifest(path)


def assert_sample_error(inputs, code):
    manifest = write_manifest(inputs)
    with pytest.raises(ValidationError) as error:
        load_sample(manifest, manifest.samples[0])
    assert error.value.code == code
    assert str(inputs[0].parent) not in str(error.value)


def test_demo_loads_without_changing_pixels():
    manifest = load_manifest(DEMO)
    assert len(manifest.samples) == 8
    for record in manifest.samples:
        loaded = load_sample(manifest, record)
        assert loaded.original.shape == loaded.edited.shape == (256, 256, 3)
        assert loaded.original.dtype == loaded.edited.dtype == np.uint8
        assert loaded.edit_mask.dtype == bool
        with Image.open(DEMO.parent / record["edited_path"]) as expected:
            np.testing.assert_array_equal(loaded.edited, np.asarray(expected))
        assert loaded.edit_mask.sum() == (
            65536 if record["sample_id"] == "no-preservation-support" else 6400
        )


@pytest.mark.parametrize(
    "text",
    [
        '{"schema_version":"1.0","schema_version":"1.0"}',
        '{"value":NaN}',
        '{"value":Infinity}',
        '{"value":-Infinity}',
        "[]",
        "{",
    ],
)
def test_invalid_json_and_structure(inputs, text):
    inputs[0].write_text(text, encoding="utf-8")
    with pytest.raises(ValidationError) as error:
        load_manifest(inputs[0])
    assert error.value.code == "invalid_manifest"


@pytest.mark.parametrize(
    "mutation",
    [
        lambda d: d.update(schema_version="2.0"),
        lambda d: d.update(samples=[]),
        lambda d: d.update(extra=True),
        lambda d: d.update(dataset_id="bad id"),
        lambda d: d["samples"].append(copy.deepcopy(d["samples"][0])),
        lambda d: d["samples"][0].pop("mask_path"),
        lambda d: d["samples"][0].update(instruction=" "),
        lambda d: d["samples"][0].update(sample_kind="unknown"),
        lambda d: d["samples"][0]["provenance"].update(generator={"model": "x", "version": "1"}),
        lambda d: d["samples"][0]["provenance"].update(
            source_url="https://user:secret@example.org"
        ),
        lambda d: d["samples"][0]["provenance"].update(extra=0),
    ],
)
def test_invalid_manifest_fields(inputs, mutation):
    mutation(inputs[1])
    with pytest.raises(ValidationError) as error:
        write_manifest(inputs)
    assert error.value.code == "invalid_manifest"


@pytest.mark.parametrize(
    "value",
    [
        "../escape.png",
        "/absolute.png",
        "C:/private.png",
        "C:private.png",
        "https://example.org/a.png",
        "assets\\mask.png",
        "\u0000.png",
    ],
)
def test_unsafe_paths(inputs, value):
    inputs[1]["samples"][0]["mask_path"] = value
    with pytest.raises(ValidationError) as error:
        write_manifest(inputs)
    assert error.value.code == "invalid_manifest"
    assert value not in str(error.value)


def test_symlink_escape(inputs):
    path = inputs[0].parent / "escape"
    try:
        path.symlink_to(inputs[0].parent.parent, target_is_directory=True)
    except OSError:
        pytest.skip("Windows symlink creation is not enabled for this account.")
    inputs[1]["samples"][0]["mask_path"] = "escape/mask.png"
    with pytest.raises(ValidationError):
        write_manifest(inputs)


def test_generated_output_unknown_metadata(inputs):
    sample = inputs[1]["samples"][0]
    sample.update(sample_kind="generated_output")
    sample["provenance"].update(generator={"model": None, "version": None}, license=None)
    manifest = write_manifest(inputs)
    loaded = load_sample(manifest, manifest.samples[0])
    assert [warning["code"] for warning in loaded.warnings] == ["unknown_provenance"]


@pytest.mark.parametrize("field", ["original_path", "edited_path", "mask_path"])
def test_missing_asset_is_sample_error(inputs, field):
    inputs[1]["samples"][0][field] = "not-here.png"
    assert_sample_error(inputs, "missing_asset")


def test_missing_local_license_is_sample_error(inputs):
    (inputs[0].parent / "LICENSE.txt").unlink()
    assert_sample_error(inputs, "missing_asset")


def test_grayscale_expands_without_value_changes(inputs):
    Image.new("L", (8, 8), 42).save(inputs[0].parent / "original.png")
    manifest = write_manifest(inputs)
    loaded = load_sample(manifest, manifest.samples[0])
    assert loaded.input_modes["original_path"] == "L"
    np.testing.assert_array_equal(loaded.original, np.full((8, 8, 3), 42, dtype=np.uint8))
    assert not loaded.warnings


@pytest.mark.parametrize("fill,empty", [(0, False), (255, True)])
def test_empty_and_full_edit_masks(inputs, fill, empty):
    Image.new("L", (8, 8), fill).save(inputs[0].parent / "mask.png")
    manifest = write_manifest(inputs)
    loaded = load_sample(manifest, manifest.samples[0])
    assert loaded.edit_mask.all() == empty
    assert bool(loaded.warnings) == empty
    if empty:
        assert loaded.warnings[0]["code"] == "empty_preservation_region"


@pytest.mark.parametrize("value", [1, 127, 254])
def test_nonbinary_mask_rejected(inputs, value):
    Image.new("L", (8, 8), value).save(inputs[0].parent / "mask.png")
    assert_sample_error(inputs, "invalid_mask")


def test_dimension_mismatch_not_resized(inputs):
    Image.new("L", (7, 8), 0).save(inputs[0].parent / "mask.png")
    assert_sample_error(inputs, "dimension_mismatch")


@pytest.mark.parametrize("mode", ["RGBA", "P", "I;16"])
def test_unsupported_modes(inputs, mode):
    Image.new(mode, (8, 8)).save(inputs[0].parent / "original.png")
    assert_sample_error(inputs, "unsupported_input")


def test_rgb_mask_rejected(inputs):
    Image.new("RGB", (8, 8)).save(inputs[0].parent / "mask.png")
    assert_sample_error(inputs, "unsupported_input")


@pytest.mark.parametrize(
    "field,code", [("original.png", "invalid_image"), ("mask.png", "invalid_mask")]
)
def test_corrupt_file(inputs, field, code):
    (inputs[0].parent / field).write_bytes(b"not an image")
    assert_sample_error(inputs, code)


def test_truncated_png(inputs):
    path = inputs[0].parent / "original.png"
    path.write_bytes(path.read_bytes()[:-20])
    assert_sample_error(inputs, "invalid_image")


def test_jpeg_with_png_extension(inputs):
    Image.new("RGB", (8, 8)).save(inputs[0].parent / "original.png", format="JPEG")
    assert_sample_error(inputs, "unsupported_input")


@pytest.mark.parametrize(
    "settings",
    [
        {"transparency": (10, 20, 30)},
        {"icc_profile": b"uninterpreted-profile"},
        {"exif": Image.Exif()},
    ],
)
def test_metadata_rejected(inputs, settings):
    if "exif" in settings:
        settings["exif"][274] = 6
    Image.new("RGB", (8, 8), (10, 20, 30)).save(inputs[0].parent / "original.png", **settings)
    assert_sample_error(inputs, "unsupported_input")


@pytest.mark.parametrize("gamma,allowed", [(45455, True), (100000, False)])
def test_declared_gamma(inputs, gamma, allowed):
    info = PngImagePlugin.PngInfo()
    info.add(b"gAMA", struct.pack(">I", gamma))
    Image.new("RGB", (8, 8)).save(inputs[0].parent / "original.png", pnginfo=info)
    if allowed:
        manifest = write_manifest(inputs)
        load_sample(manifest, manifest.samples[0])
    else:
        assert_sample_error(inputs, "unsupported_input")


@pytest.mark.parametrize("size", [(4097, 1), (2049, 2048)])
def test_oversized_before_pixel_decode(inputs, monkeypatch, size):
    Image.new("L", size).save(inputs[0].parent / "original.png")

    def unexpected_decode(self):
        pytest.fail("Oversized input must be rejected before pixel decoding.")

    monkeypatch.setattr(PngImagePlugin.PngImageFile, "load", unexpected_decode)
    assert_sample_error(inputs, "input_limit_exceeded")


def test_animated_png_rejected(inputs):
    first = Image.new("RGB", (8, 8), "red")
    first.save(
        inputs[0].parent / "original.png",
        save_all=True,
        append_images=[Image.new("RGB", (8, 8), "blue")],
        duration=100,
    )
    assert_sample_error(inputs, "unsupported_input")


def test_sixteen_bit_rgb_is_not_silently_quantized(inputs):
    def chunk(kind, data):
        return (
            struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))
        )

    png = b"\x89PNG\r\n\x1a\n"
    png += chunk(b"IHDR", struct.pack(">IIBBBBB", 8, 8, 16, 2, 0, 0, 0))
    png += chunk(b"IDAT", zlib.compress((b"\x00" + b"\x00\x80" * 24) * 8))
    png += chunk(b"IEND", b"")
    (inputs[0].parent / "original.png").write_bytes(png)
    assert_sample_error(inputs, "unsupported_input")


@pytest.mark.parametrize(
    "kind,data,allowed",
    [
        (b"sRGB", b"\x00", True),
        (b"sRGB", b"\x04", False),
        (b"cHRM", struct.pack(">8I", 31270, 32900, 64000, 33000, 30000, 60000, 15000, 6000), True),
        (b"cHRM", struct.pack(">8I", *([10000] * 8)), False),
        (b"cICP", bytes([1, 13, 0, 1]), False),
    ],
)
def test_color_declarations(inputs, kind, data, allowed):
    info = PngImagePlugin.PngInfo()
    info.add(kind, data)
    Image.new("RGB", (8, 8)).save(inputs[0].parent / "original.png", pnginfo=info)
    if allowed:
        manifest = write_manifest(inputs)
        load_sample(manifest, manifest.samples[0])
    else:
        assert_sample_error(inputs, "unsupported_input")
