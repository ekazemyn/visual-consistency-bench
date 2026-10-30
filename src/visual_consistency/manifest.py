"""Strict manifest validation and contained local-path resolution."""

import json
import re
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from urllib.parse import urlsplit

IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
SAMPLE_FIELDS = {
    "sample_id",
    "source_group_id",
    "original_path",
    "edited_path",
    "mask_path",
    "instruction",
    "sample_kind",
    "provenance",
}
PROVENANCE_FIELDS = {"source", "source_url", "license", "license_reference", "generator"}


class ValidationError(ValueError):
    """A validation failure with a stable code and path-free message."""

    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(message)


@dataclass
class Manifest:
    """Validated manifest structure; asset existence is checked when samples load."""

    path: Path
    dataset_id: str
    samples: list[dict]


def _invalid(message):
    raise ValidationError("invalid_manifest", message)


def _fields(value, expected, label):
    if not isinstance(value, dict) or set(value) != expected:
        _invalid(f"{label} must contain exactly its documented fields.")


def _text(value, label, nullable=False):
    if nullable and value is None:
        return
    if not isinstance(value, str) or not value.strip():
        _invalid(f"{label} must be a nonempty string.")


def _identifier(value, label):
    if not isinstance(value, str) or not IDENTIFIER.fullmatch(value):
        _invalid(f"{label} must be a valid identifier of at most 128 characters.")


def _url(value, label):
    _text(value, label)
    try:
        parts = urlsplit(value)
        valid = (
            parts.scheme in {"http", "https"}
            and parts.hostname
            and not parts.username
            and not parts.password
            and not any(character.isspace() for character in value)
        )
        parts.port
    except ValueError:
        valid = False
    if not valid:
        _invalid(f"{label} must be a public HTTP(S) URL without credentials.")


def resolve_local_path(root: Path, value: str, label: str) -> Path:
    """Resolve a portable relative path, including symlinks, without leaving root."""
    _text(value, label)
    try:
        windows = PureWindowsPath(value)
        if (
            "\\" in value
            or ":" in value
            or "\x00" in value
            or windows.drive
            or windows.root
            or Path(value).is_absolute()
        ):
            _invalid(f"{label} must be a local relative path using forward slashes.")
        base = root.resolve()
        path = (base / value).resolve()
        if not path.is_relative_to(base):
            _invalid(f"{label} must remain inside the manifest directory.")
        return path
    except (OSError, RuntimeError, ValueError) as error:
        if isinstance(error, ValidationError):
            raise
        _invalid(f"{label} could not be resolved inside the manifest directory.")


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            _invalid("Duplicate JSON object keys are not allowed.")
        result[key] = value
    return result


def _constant(value):
    _invalid("Non-finite JSON numbers are not allowed.")


def load_manifest(path: str | Path) -> Manifest:
    """Validate structure and path containment without loading image pixels."""
    path = Path(path)
    try:
        data = json.loads(
            path.read_text(encoding="utf-8"), object_pairs_hook=_object, parse_constant=_constant
        )
    except (OSError, UnicodeError, json.JSONDecodeError):
        raise ValidationError(
            "invalid_manifest", "Manifest must be a readable UTF-8 JSON file."
        ) from None
    _fields(data, {"schema_version", "dataset_id", "samples"}, "Manifest")
    if data["schema_version"] != "1.0":
        _invalid("Unsupported manifest schema_version; expected 1.0.")
    _identifier(data["dataset_id"], "dataset_id")
    if not isinstance(data["samples"], list) or not data["samples"]:
        _invalid("samples must be a nonempty array.")
    root = path.resolve().parent
    seen = set()
    for sample in data["samples"]:
        _fields(sample, SAMPLE_FIELDS, "Sample")
        _identifier(sample["sample_id"], "sample_id")
        _identifier(sample["source_group_id"], "source_group_id")
        if sample["sample_id"] in seen:
            _invalid("sample_id values must be unique within a manifest.")
        seen.add(sample["sample_id"])
        _text(sample["instruction"], "instruction")
        if sample["sample_kind"] not in ("synthetic_control", "generated_output"):
            _invalid("sample_kind must be synthetic_control or generated_output.")
        for field in ("original_path", "edited_path", "mask_path"):
            resolve_local_path(root, sample[field], field)
        provenance = sample["provenance"]
        _fields(provenance, PROVENANCE_FIELDS, "provenance")
        _text(provenance["source"], "source")
        _text(provenance["license"], "license", nullable=True)
        if provenance["source_url"] is not None:
            _url(provenance["source_url"], "source_url")
        reference = provenance["license_reference"]
        if reference is not None:
            _text(reference, "license_reference")
            if reference.lower().startswith(("http:", "https:")):
                _url(reference, "license_reference")
            else:
                resolve_local_path(root, reference, "license_reference")
        generator = provenance["generator"]
        if sample["sample_kind"] == "synthetic_control":
            if generator is not None:
                _invalid("Synthetic controls must have null generator metadata.")
        else:
            _fields(generator, {"model", "version"}, "generator")
            for field in ("model", "version"):
                _text(generator[field], f"generator.{field}", nullable=True)
    return Manifest(path.resolve(), data["dataset_id"], data["samples"])
