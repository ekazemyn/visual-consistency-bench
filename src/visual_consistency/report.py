"""Self-contained, bounded HTML inspection reports with escaped supplied text."""

import base64
import hashlib
import html
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image

from visual_consistency.images import load_sample
from visual_consistency.manifest import ValidationError, resolve_local_path

MAX_PREVIEW_SIDE = 256
MAX_PREVIEW_BYTES = 4 * 1024 * 1024
MAX_REPORT_BYTES = 8 * 1024 * 1024

STYLE = """
:root{color-scheme:light;--ink:#162b39;--muted:#526775;--line:#d9e2e8}
*{box-sizing:border-box}body{margin:0;background:#eef3f6;color:var(--ink);
font:16px/1.55 system-ui,sans-serif}main{max-width:1200px;margin:auto;padding:32px 24px}
header{padding:24px 0}h1{font-size:clamp(2rem,4vw,3rem);line-height:1.1;margin:12px 0}
h2{margin:0;font-size:1.35rem}h3{font-size:1rem;margin-bottom:8px}
p{margin:10px 0}.eyebrow{color:#126b70;font-weight:700;letter-spacing:.12em;
text-transform:uppercase;font-size:.8rem}.muted,figcaption{color:var(--muted)}
.summary{display:flex;gap:16px;flex-wrap:wrap;margin:24px 0}.summary div{background:white;
border:1px solid var(--line);border-radius:12px;padding:12px 24px;min-width:140px}
.summary strong{display:block;font-size:1.8rem}.guide,article{background:white;
border:1px solid var(--line);border-radius:16px;padding:24px;margin:24px 0}
.tag{display:inline-block;background:#e4eff2;color:#244c57;padding:3px 10px;
border-radius:20px;font-size:.8rem;margin:8px 8px 4px 0}.failure{background:#fff0eb;
border-left:4px solid #9a4429;padding:12px}.notice{background:#f5f7f9;padding:12px}
.previews{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px;margin:20px 0}
figure{margin:0;min-width:0}figure img{display:block;width:100%;height:240px;
object-fit:contain;background:#e9eef2;border:1px solid var(--line);border-radius:8px}
figcaption{font-size:.8rem;margin-top:6px}figcaption strong{display:block;color:var(--ink)}
.table-wrap{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:.9rem}
th,td{text-align:left;padding:10px;border-bottom:1px solid var(--line);vertical-align:top}
th{color:var(--muted)}pre{white-space:pre-wrap;font:inherit;font-size:.85rem}
p,td,pre,figcaption,h2{overflow-wrap:anywhere}details{margin-top:16px}
footer{color:var(--muted);font-size:.85rem;margin:32px 0}
@media(max-width:900px){.previews{grid-template-columns:repeat(2,minmax(0,1fr))}}
@media(max-width:520px){main{padding:16px}article,.guide{padding:16px}
.previews{grid-template-columns:1fr}figure img{height:auto;max-height:256px}}
"""


def _text(value):
    text = str(value)
    if len(text) > 4000:
        text = text[:4000] + "… [text shortened; full text in results.json]"
    return html.escape(text, quote=True)


def _preview_panels(loaded):
    original = loaded.original.astype(np.float64)
    overlay = original.copy()
    overlay[loaded.edit_mask] = 0.6 * original[loaded.edit_mask] + 0.4 * np.array([255, 0, 255])
    difference = np.abs(loaded.edited.astype(np.float64) - original).mean(axis=2)
    return (
        Image.fromarray(loaded.original),
        Image.fromarray(loaded.edited),
        Image.fromarray(np.rint(overlay).astype(np.uint8)),
        Image.fromarray(np.rint(difference).astype(np.uint8)),
    )


def _previews(manifest, record, sample, budget):
    if sample["status"] != "ok":
        return '<p class="failure">Previews unavailable: sample evaluation failed.</p>', 0
    if budget <= 0:
        return '<p class="notice">Preview budget reached; measurements retained.</p>', 0
    try:
        loaded = load_sample(manifest, record)
        # Reopening assets must not associate changed pixels with old measurements.
        for field, expected in sample["input_sha256"].items():
            path = resolve_local_path(manifest.path.parent, record[field], field)
            with path.open("rb") as source:
                if hashlib.file_digest(source, "sha256").hexdigest() != expected:
                    return '<p class="failure">Previews unavailable: input bytes changed.</p>', 0
    except (ValidationError, OSError):
        return '<p class="failure">Previews unavailable: inputs could not be reloaded.</p>', 0
    captions = (
        ("Original", Path(record["original_path"]).name),
        ("Edited", Path(record["edited_path"]).name),
        ("Edit mask overlay", "Magenta tint = permitted edit; untinted = preservation."),
        ("Absolute difference", "Mean RGB |edit − original| / 255. Black = 0; white = 1."),
    )
    panels = []
    used = 0
    for image, (title, caption) in zip(_preview_panels(loaded), captions):
        image.thumbnail((MAX_PREVIEW_SIDE, MAX_PREVIEW_SIDE), Image.Resampling.BILINEAR)
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        used += len(encoded)
        if used > budget:
            return '<p class="notice">Preview budget reached; measurements retained.</p>', 0
        panels.append(
            f'<figure><img src="data:image/png;base64,{encoded}" '
            f'alt="{title} preview for {_text(sample["sample_id"])}" '
            f'width="{image.width}" height="{image.height}">'
            f"<figcaption><strong>{title}</strong>{_text(caption)}</figcaption></figure>"
        )
    return '<div class="previews">' + "".join(panels) + "</div>", used


def _metrics(sample):
    if not sample["metrics"]:
        return '<p class="failure">No measurements: sample failed. See errors below.</p>'
    labels = {
        "preservation_mae": "Preservation MAE",
        "preservation_psnr": "Preservation PSNR (dB)",
        "preservation_ssim": "Preservation SSIM",
        "inside_mask_mae": "Inside-mask MAE",
    }
    rows = []
    for name, metric in sample["metrics"].items():
        value = metric["value"]
        display = f"{value:.6g}" if value is not None else "—"
        if metric["status"] == "perfect_match":
            display = "Perfect match (unbounded)"
        rows.append(
            f'<tr><th scope="row">{_text(labels[name])}</th><td>{display}</td>'
            f"<td>{_text(metric['status'])}</td><td>{metric['support_count']:,} "
            f"{_text(metric['support_unit'])}</td><td>{_text(metric['direction'])}</td></tr>"
        )
    return (
        '<div class="table-wrap"><table><caption>Measurements and applicable support</caption>'
        '<thead><tr><th scope="col">Metric</th><th scope="col">Value</th>'
        '<th scope="col">Status</th><th scope="col">Support</th>'
        '<th scope="col">Direction</th></tr></thead><tbody>'
        + "".join(rows)
        + "</tbody></table></div>"
    )


def render_report(result, manifest):
    """Render evaluated records with sequential preview loading; never fetch URLs."""
    counts = result["counts"]
    parts = [
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        '<meta http-equiv="Content-Security-Policy" content="default-src &#39;none&#39;; '
        "img-src data:; style-src &#39;unsafe-inline&#39;; base-uri &#39;none&#39;; "
        'form-action &#39;none&#39;">'
        f"<title>Image preservation · {_text(result['dataset_id'])}</title>"
        f"<style>{STYLE}</style></head><body><main><header>"
        '<p class="eyebrow">Image edit inspection</p><h1>What changed outside the edit?</h1>'
        f'<p class="muted">Dataset: {_text(result["dataset_id"])}</p>'
        "<p>Pixel preservation and instruction success are separate questions. "
        "An unchanged image can preserve every pixel while failing the instruction.</p></header>"
        '<div class="summary" aria-label="Run counts">'
        + "".join(
            f"<div><strong>{counts[key]:,}</strong>{key.title()}</div>"
            for key in ("submitted", "evaluated", "failed")
        )
        + '</div><section class="guide" aria-label="How to read this report">'
        "<h2>Read the evidence</h2><p>MAE: lower means less pixel change (range 0–1). "
        "PSNR: higher means less pixel change; exact matches are unbounded. "
        "SSIM: higher means greater local similarity; values can be negative. "
        "SSIM averages RGB over complete uniform 7×7 preservation windows only.</p>"
        "<p>Inside-mask MAE describes change, with no preferred direction. "
        "Not applicable means no valid support, never a perfect score. "
        "There is no combined quality score or automatic instruction-success judgment.</p>"
        "<p>All four previews use the same scale, at most 256 pixels per side. "
        "Only previews are downsampled. The difference uses a fixed 0–1 grayscale "
        "scale across every sample, includes both regions, and is never contrast-normalized.</p>"
        "</section>"
    ]
    size = len(parts[0].encode("utf-8"))
    preview_bytes = 0
    for record, sample in zip(manifest.samples, result["samples"]):
        previews, used = _previews(manifest, record, sample, MAX_PREVIEW_BYTES - preview_bytes)
        preview_bytes += used
        support = sample["support"]
        support_text = (
            f"{support['preservation_pixels']:,} preservation / "
            f"{support['permitted_edit_pixels']:,} permitted edit / "
            f"{support['total_pixels']:,} total pixels; "
            f"{support['ssim_windows']:,} valid SSIM window locations."
            if support is not None
            else "Pixel support unavailable."
        )
        diagnostics = "".join(
            f'<p class="{"failure" if kind == "errors" else "notice"}">'
            f"{kind.title()}: <strong>{_text(item['code'])}</strong> — {_text(item['message'])}</p>"
            for kind in ("errors", "warnings")
            for item in sample[kind]
        )
        provenance = json.dumps(sample["provenance"], indent=2, ensure_ascii=False)
        section = (
            f"<article><h2>{_text(sample['sample_id'])}</h2>"
            f'<span class="tag">{_text(sample["sample_kind"])}</span>'
            f'<span class="tag">Status: {_text(sample["status"])}</span>'
            f'<p class="muted">Source group: {_text(sample["source_group_id"])}</p>'
            f"<h3>Instruction</h3><p>{_text(sample['instruction'])}</p>"
            f'{previews}<p class="muted">{support_text}</p>{_metrics(sample)}{diagnostics}'
            f"<details><summary>Provenance supplied by the dataset</summary>"
            f"<pre>{_text(provenance)}</pre></details></article>"
        )
        size += len(section.encode("utf-8"))
        if size > MAX_REPORT_BYTES - 2048:
            raise ValidationError(
                "report_limit_exceeded", "HTML exceeds the 8 MiB limit; full JSON/CSV retained."
            )
        parts.append(section)
    parts.append(
        "<footer>Standalone offline report. No external assets, scripts, or fonts. "
        f"Contract {_text(result['contract_version'])} · "
        f"Package {_text(result['package_version'])} · "
        f"UTC {_text(result['started_at_utc'])}<br>"
        f"Manifest SHA-256: {_text(result['manifest_sha256'])}<br>"
        "Full precision, hashes, settings, timing, and versions are in results.json; "
        "tidy metric records are in metrics.csv. Review supplied text and image rights "
        "before sharing.</footer></main></body></html>"
    )
    return "".join(parts)
