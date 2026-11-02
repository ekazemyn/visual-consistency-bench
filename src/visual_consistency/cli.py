"""Command-line image evaluation and portable structured results."""

import argparse
import csv
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

from visual_consistency.images import MAX_PIXELS, MAX_SIDE, load_sample
from visual_consistency.manifest import ValidationError, load_manifest, resolve_local_path
from visual_consistency.metrics import SSIM_PARAMETERS, measure_preservation
from visual_consistency.report import render_report

ASSETS = ("original_path", "edited_path", "mask_path")
CSV_FIELDS = (
    "sample_id",
    "source_group_id",
    "sample_kind",
    "sample_status",
    "metric",
    "value",
    "metric_status",
    "support_count",
    "support_unit",
    "direction",
    "parameters",
    "warning_codes",
    "error_codes",
)


def _sha256(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def evaluate_manifest(manifest):
    """Evaluate in manifest order, retaining sample failures without private paths."""
    started = datetime.now(timezone.utc).isoformat()
    start = perf_counter()
    manifest_hash = _sha256(manifest.path)
    samples = []
    for record in manifest.samples:
        sample_start = perf_counter()
        result = {
            key: record[key]
            for key in ("sample_id", "source_group_id", "sample_kind", "instruction", "provenance")
        }
        result.update(
            status="error",
            input_sha256=dict.fromkeys(ASSETS),
            width=None,
            height=None,
            input_modes=None,
            grayscale_expanded=None,
            support=None,
            metrics={},
            warnings=[],
            errors=[],
        )
        try:
            for field in ASSETS:
                path = resolve_local_path(manifest.path.parent, record[field], field)
                try:
                    result["input_sha256"][field] = _sha256(path)
                except OSError:
                    raise ValidationError(
                        "missing_asset", f"{field} must refer to a readable file."
                    ) from None
            loaded = load_sample(manifest, record)
            result.update(
                width=int(loaded.original.shape[1]),
                height=int(loaded.original.shape[0]),
                input_modes=loaded.input_modes,
                grayscale_expanded={
                    field: loaded.input_modes[field] == "L" for field in ASSETS[:2]
                },
                warnings=loaded.warnings.copy(),
            )
            measurements = measure_preservation(loaded)
            result.update(support=measurements["support"], metrics=measurements["metrics"])
            result["warnings"].extend(measurements["warnings"])
            # A non-finite measurement is a failed sample, never nonstandard JSON.
            json.dumps(result, allow_nan=False)
            result["status"] = "ok"
        except ValidationError as error:
            result["errors"].append({"code": error.code, "message": str(error)})
        except Exception:
            result["metrics"] = {}
            result["errors"].append(
                {"code": "processing_error", "message": "Sample processing failed."}
            )
        result["wall_duration_seconds"] = perf_counter() - sample_start
        samples.append(result)
    evaluated = sum(sample["status"] == "ok" for sample in samples)
    return {
        "schema_version": "1.0",
        "contract_version": "1.0",
        "package_version": version("visual-consistency"),
        "dependency_versions": {
            name: version(name) for name in ("numpy", "scipy", "Pillow", "scikit-image")
        },
        "python_version": platform.python_version(),
        "dataset_id": manifest.dataset_id,
        "manifest_sha256": manifest_hash,
        "started_at_utc": started,
        "wall_duration_seconds": perf_counter() - start,
        "timing_definition": (
            "Manifest hashing, sample hashing/loading/measurement, and metadata collection; "
            "excludes manifest validation and output writing."
        ),
        "settings": {
            "max_side": MAX_SIDE,
            "max_pixels": MAX_PIXELS,
            "normalization": "rgb_bytes/255",
            "alignment": "as_supplied",
            "ssim": SSIM_PARAMETERS.copy(),
        },
        "counts": {
            "submitted": len(samples),
            "evaluated": evaluated,
            "failed": len(samples) - evaluated,
        },
        "samples": samples,
    }


def write_results(result, output):
    """Write strict JSON and tidy CSV, including a row for every failed sample."""
    encoded = json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    (output / "results.json").write_text(encoded, encoding="utf-8")
    with (output / "metrics.csv").open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for sample in result["samples"]:
            base = {key: sample[key] for key in ("sample_id", "source_group_id", "sample_kind")}
            base.update(
                sample_status=sample["status"],
                warning_codes=";".join(item["code"] for item in sample["warnings"]),
                error_codes=";".join(item["code"] for item in sample["errors"]),
            )
            for name, metric in sample["metrics"].items():
                writer.writerow(
                    {
                        **base,
                        "metric": name,
                        "value": metric["value"],
                        "metric_status": metric["status"],
                        "support_count": metric["support_count"],
                        "support_unit": metric["support_unit"],
                        "direction": metric["direction"],
                        "parameters": json.dumps(
                            metric["parameters"], sort_keys=True, allow_nan=False
                        ),
                    }
                )
            if not sample["metrics"]:
                writer.writerow(base)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Measure image-edit preservation from a manifest.")
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path, help="Directory for JSON, CSV, and HTML results")
    parser.add_argument("--overwrite", action="store_true", help="Replace the three result files")
    args = parser.parse_args(argv)
    try:
        manifest = load_manifest(args.manifest)
        targets = [args.output / name for name in ("results.json", "metrics.csv", "report.html")]
        protected = {manifest.path}
        for record in manifest.samples:
            for field in ASSETS:
                protected.add(resolve_local_path(manifest.path.parent, record[field], field))
            reference = record["provenance"]["license_reference"]
            if reference is not None and not reference.lower().startswith(("http:", "https:")):
                protected.add(
                    resolve_local_path(manifest.path.parent, reference, "license_reference")
                )
        for target in targets:
            if target.is_symlink() or target.resolve() in protected:
                raise ValidationError(
                    "output_conflict", "Output files must not replace input assets or symlinks."
                )
            if target.exists() and not args.overwrite:
                raise ValidationError(
                    "output_exists", "Results already exist; use --overwrite to replace them."
                )
        result = evaluate_manifest(manifest)
        args.output.mkdir(parents=True, exist_ok=True)
        write_results(result, args.output)
        (args.output / "report.html").write_text(render_report(result, manifest), encoding="utf-8")
    except ValidationError as error:
        print(f"{error.code}: {error}", file=sys.stderr)
        return 2
    except OSError:
        print("io_error: Could not read inputs or write output files.", file=sys.stderr)
        return 2
    counts = result["counts"]
    print(
        f"Evaluated {counts['evaluated']}/{counts['submitted']} samples; {counts['failed']} failed."
    )
    return 1 if counts["failed"] else 0
