"""Bounded, non-transforming PNG loading for image preservation inputs."""

import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from visual_consistency.manifest import Manifest, ValidationError, resolve_local_path

MAX_SIDE = 4096
MAX_PIXELS = 4_194_304
SRGB_CHROMATICITY = (0.3127, 0.3290, 0.64, 0.33, 0.30, 0.60, 0.15, 0.06)


@dataclass
class LoadedSample:
    """RGB byte arrays and a boolean edit mask; True permits editing."""

    record: dict
    original: np.ndarray
    edited: np.ndarray
    edit_mask: np.ndarray
    input_modes: dict[str, str]
    warnings: list[dict[str, str]]


def _unsupported(message):
    raise ValidationError("unsupported_input", message)


def _check_png_encoding(path, label):
    # Pillow can expose 16-bit RGB PNGs as RGB bytes, so inspect the source depth.
    # verify() has already checked chunk integrity; skip payloads without decoding them.
    with path.open("rb") as source:
        header = source.read(29)
        if len(header) != 29 or header[24] != 8:
            _unsupported(f"{label} must use 8-bit PNG samples.")
        source.seek(4, 1)  # IHDR checksum
        while chunk := source.read(8):
            if len(chunk) != 8:
                raise ValueError("Truncated PNG chunk")
            length, kind = int.from_bytes(chunk[:4], "big"), chunk[4:]
            if kind in (b"cICP", b"mDCV", b"cLLI"):
                _unsupported(f"{label} declares unsupported advanced color metadata.")
            if kind == b"IEND":
                break
            source.seek(length + 4, 1)


def _read_png(path: Path, label: str, mask=False):
    if not path.is_file():
        raise ValidationError("missing_asset", f"{label} must refer to an existing file.")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(path) as image:
                width, height = image.size
                if (
                    not (0 < width <= MAX_SIDE and 0 < height <= MAX_SIDE)
                    or width * height > MAX_PIXELS
                ):
                    raise ValidationError(
                        "input_limit_exceeded", f"{label} exceeds image dimensions or pixel limit."
                    )
                if image.format != "PNG" or image.n_frames != 1:
                    _unsupported(f"{label} must be a single-frame PNG.")
                if image.mode not in (("L",) if mask else ("RGB", "L")):
                    _unsupported(f"{label} has an unsupported pixel mode.")
                image.verify()
            _check_png_encoding(path, label)
            with Image.open(path) as image:
                image.load()
                if "transparency" in image.info or "icc_profile" in image.info:
                    _unsupported(f"{label} must not declare transparency or an ICC profile.")
                if image.getexif().get(274, 1) != 1:
                    _unsupported(f"{label} must have absent or identity EXIF orientation.")
                gamma = image.info.get("gamma")
                chromaticity = image.info.get("chromaticity")
                if gamma is not None and abs(gamma - 0.45455) > 0.00001:
                    _unsupported(f"{label} declares a conflicting gamma value.")
                if chromaticity is not None and not np.allclose(
                    chromaticity, SRGB_CHROMATICITY, rtol=0, atol=0.00001
                ):
                    _unsupported(f"{label} declares conflicting chromaticities.")
                if "srgb" in image.info and image.info["srgb"] not in (0, 1, 2, 3):
                    _unsupported(f"{label} declares an invalid sRGB rendering intent.")
                mode = image.mode
                pixels = np.array(image if mask else image.convert("RGB"), dtype=np.uint8)
    except (Image.DecompressionBombError, Image.DecompressionBombWarning):
        raise ValidationError(
            "input_limit_exceeded", f"{label} exceeds safe decode limits."
        ) from None
    except (OSError, ValueError, SyntaxError) as error:
        if isinstance(error, ValidationError):
            raise
        code = "invalid_mask" if mask else "invalid_image"
        raise ValidationError(code, f"{label} could not be decoded as a valid PNG.") from None
    if mask and not np.all((pixels == 0) | (pixels == 255)):
        raise ValidationError("invalid_mask", "mask_path must contain only byte values 0 and 255.")
    return pixels, mode


def load_sample(manifest: Manifest, sample: dict) -> LoadedSample:
    """Load one validated record; callers handle sample errors without losing its ID."""
    root = manifest.path.parent
    provenance = sample["provenance"]
    reference = provenance["license_reference"]
    if reference is not None and not reference.lower().startswith(("http:", "https:")):
        notice = resolve_local_path(root, reference, "license_reference")
        if not notice.is_file():
            raise ValidationError(
                "missing_asset", "Local license_reference must refer to an existing file."
            )
    arrays = {}
    modes = {}
    for field in ("original_path", "edited_path", "mask_path"):
        path = resolve_local_path(root, sample[field], field)
        arrays[field], modes[field] = _read_png(path, field, mask=field == "mask_path")
    original, edited, mask = (
        arrays[field] for field in ("original_path", "edited_path", "mask_path")
    )
    if original.shape[:2] != edited.shape[:2] or original.shape[:2] != mask.shape:
        raise ValidationError(
            "dimension_mismatch", "Original, edit, and mask must have identical dimensions."
        )
    diagnostics = []
    generator = provenance["generator"]
    if (
        any(provenance[field] is None for field in ("source_url", "license", "license_reference"))
        or generator is not None
        and any(value is None for value in generator.values())
    ):
        diagnostics.append(
            {
                "code": "unknown_provenance",
                "message": "Provenance is incomplete; no redistribution rights are implied.",
            }
        )
    edit_mask = mask == 255
    if edit_mask.all():
        diagnostics.append(
            {
                "code": "empty_preservation_region",
                "message": "No preservation pixels are available.",
            }
        )
    return LoadedSample(sample, original, edited, edit_mask, modes, diagnostics)
