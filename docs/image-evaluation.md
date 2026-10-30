# Image preservation evaluation contract

Contract version: `1.0`.

This document specifies inputs, validation, and result semantics for evaluating changes
outside a permitted image-edit region. It specifies behavior without claiming that
the measurements agree with human judgments. Manifest and asset loading
are available through the Python API below; metric evaluation is not implemented.

## Load and validate inputs

From Python with the project installed, run from the repository root:

```python
from visual_consistency.images import load_sample
from visual_consistency.manifest import ValidationError, load_manifest

manifest = load_manifest("examples/images/manifest.json")
for record in manifest.samples:
    try:
        loaded = load_sample(manifest, record)
    except ValidationError as error:
        print(record["sample_id"], error.code, str(error))
        continue
    print(record["sample_id"], loaded.original.shape, loaded.warnings)
```

`load_manifest` checks JSON structure, metadata, identifiers, duplicate sample IDs,
and contained paths. Structural errors raise `ValidationError` with code
`invalid_manifest` before samples load. File existence, decode validity, local license
notice existence, and matching dimensions are checked by `load_sample`; one bad asset
does not prevent a caller from inspecting subsequent records. URLs are never fetched.

`LoadedSample.original` and `.edited` are RGB `uint8` arrays, not normalized floats.
`.edit_mask` is a two-dimensional boolean array: True permits editing. `.input_modes`
records original Pillow modes, including grayscale conversion. `.record` retains the
validated sample metadata, and `.warnings` contains code/message objects. No resampling,
metrics, or automatic editing-success judgments occur. An all-255 mask loads with an
`empty_preservation_region` warning. A missing source URL or license field, or unknown
generator details, produces `unknown_provenance`; it does not invalidate image pixels.

Pass records from the validated manifest unchanged to `load_sample`. It rechecks
resolved path containment when opening assets, but is not a second JSON-schema parser.

## What is measured

Given an original image, an edited image, an edit instruction, and an explicit mask,
measure how much the edited image changes outside the permitted edit region.

Preservation and instruction success are separate questions. An unchanged image can
preserve every pixel while failing an instruction. Changes inside the permitted region
do not establish that the requested edit succeeded. No combined quality score, automatic
instruction-success judgment, or universal pass/fail threshold is defined here.

## Manifest

Use UTF-8 JSON with the fields below. Field names are case-sensitive; unknown fields
are rejected for version `1.0` rather than silently ignored. Duplicate JSON object keys,
non-finite numbers, unsupported versions, and an empty sample list are invalid.

| Field | Type | Meaning |
| --- | --- | --- |
| `schema_version` | string | Exactly `"1.0"`; independent of package version |
| `dataset_id` | string | Stable identifier for this collection |
| `samples` | array of objects | Ordered, nonempty sample records |

Sample fields:

| Field | Type | Meaning |
| --- | --- | --- |
| `sample_id` | string | Unique within the manifest; do not derive from machine paths |
| `source_group_id` | string | Shared by all edits/variants of the same original source |
| `original_path` | string | Local original-image path relative to the manifest directory |
| `edited_path` | string | Local edited-image path relative to the manifest directory |
| `mask_path` | string | Local binary edit-mask path relative to the manifest directory |
| `instruction` | string | Nonempty requested edit; recorded, not automatically scored |
| `sample_kind` | string | `synthetic_control` or `generated_output` |
| `provenance` | object | Source, rights, and generator metadata below |

Identifiers must match `[A-Za-z0-9][A-Za-z0-9._-]{0,127}`. Paths and other required
strings must be nonempty after trimming whitespace. IDs stay stable when files move;
content hashes, rather than IDs, identify exact asset versions in results. Reuse
`source_group_id` across manifests when originals are shared; source groups, not
individual edits, are the unit for preventing related-sample leakage in studies.

`provenance` has exactly these required fields:

| Field | Type | Meaning |
| --- | --- | --- |
| `source` | string | Nonempty origin/creation description, without personal machine paths |
| `source_url` | string or null | Public HTTP(S) source URL if available; metadata only, never fetched |
| `license` | string or null | Stated license identifier/name if verified; null when unknown |
| `license_reference` | string or null | Relative local notice path or public HTTP(S) license URL |
| `generator` | object or null | Null for synthetic controls; generated-output object described below |

For `generated_output`, `generator` is an object with exactly `model` and `version`,
each a nonempty string or null when unknown. For `synthetic_control`, it must be null.
The instruction is the requested edit, not evidence of a known generation prompt.
Do not infer generator identity or license from appearance. Null source/license fields
are preserved as unknown and yield provenance warnings; they do not grant permission
to redistribute assets. Public examples require independently documented use rights.

Local asset paths use forward slashes and must resolve inside the manifest directory
or its descendants. Reject absolute paths, URLs in asset fields, traversal outside that
directory, and symbolic links resolving outside it. The same containment rule applies
to local license references, which must exist when supplied. Public URLs are not checked
for network reachability during evaluation. All image files must exist and decode.

See [the illustrative manifest](../examples/image-manifest.example.json). Its asset
paths are placeholders and rights metadata are explicitly unknown: it is a format
example, not a runnable dataset or a claim about existing images.

The [procedural control collection](../examples/images/README.md) provides a separate
[manifest with included assets](../examples/images/manifest.json). It demonstrates
known pixel changes; it is not evidence of performance on generated model outputs.

## Image and mask representation

### Supported input boundary

- Original and edited images: single-frame PNG with 8-bit RGB or 8-bit grayscale pixels.
  Grayscale is converted to RGB by replicating its channel; RGB bytes remain unchanged.
- Reject palette, alpha/transparency (including PNG transparency metadata), 16-bit,
  floating-point, and animated inputs. Do not silently composite, quantize, or select frames.
- Treat accepted stored RGB values as sRGB-encoded values for measurement. No linear-light
  conversion, histogram matching, exposure correction, or color-profile conversion is
  applied. Reject embedded ICC profiles to avoid silently ignoring declared color spaces.
  sRGB declarations are allowed; other explicit conflicting color declarations are invalid.
  Supported declarations are sRGB, matching gAMA (`0.45455`) and matching sRGB cHRM
  coordinates (absolute tolerance `0.00001`). Advanced cICP/mDCV/cLLI color metadata
  is unsupported and rejected rather than silently interpreted.
- Images and masks must have absent or identity EXIF orientation. Reject nonidentity
  orientation instead of independently rotating assets and risking mask misalignment.
- Require positive dimensions, at most 4,096 pixels on either side, and at most
  4,194,304 pixels per asset. These are input limits, not a measured runtime guarantee.
  Inspect dimensions before allocating full decoded arrays when possible.
- Original, edited image, and mask must have identical width and height. No resizing,
  cropping, registration, rotation, or mask interpolation occurs during evaluation.

The supplier is responsible for pixel correspondence. Equal dimensions do not prove
alignment: a shifted image is measured as supplied, not automatically corrected.
Any external preprocessing must be declared in dataset documentation.

### Mask semantics

The mask is a single-frame 8-bit grayscale PNG containing only stored values `0` and
`255`. Normalize it by dividing by 255 to obtain logical values `0` and `1`:

| Stored value | Logical value | Meaning |
| --- | --- | --- |
| `0` | `0` | Preservation region: changes are measured here |
| `255` | `1` | Permitted edit region: excluded from preservation measurements |

RGB, palette, transparent, and non-binary masks are invalid. In particular, grayscale
values `1` and `254` are not accepted or thresholded. Anti-aliased mask edges require
an explicit upstream binarization decision, never an automatic hidden threshold.

An all-zero mask means the whole image must be preserved and is valid. An all-255 mask
provides no preservation support: loading succeeds, but preservation metrics are
`not_applicable`. Mask direction is never inferred from the images or instruction.

## Measurement semantics

Use RGB channel values normalized from bytes to `[0, 1]`. Preservation MAE and MSE
are means over all three channels of pixels where logical mask equals zero. PSNR
uses data range 1 and those preservation-region MSE values. Lower MAE and higher
PSNR indicate less pixel change, not necessarily better editing.

Exact equality over all preservation pixels is a perfect preservation match. Its
PSNR is mathematically unbounded; JSON uses `value: null`, `status: "perfect_match"`
rather than Infinity. A null value with `status: "not_applicable"` means something
different and must never be ranked as a perfect match. MAE for a perfect match is 0.

If SSIM is reported, publish its window/settings and aggregate only map locations
whose entire window lies in the preservation region and inside image boundaries.
Report support in window locations separately from preservation-pixel counts. Too
little spatial support yields `not_applicable`, even when MAE/PSNR remain available.
Masked-out pixels must not be replaced with zeros to manufacture a regional SSIM.
This contract defines support policy, not a specific SSIM parameter implementation.

Any reported inside-mask distance is descriptive and labeled separately. No measurement
is normalized across a batch to create a relative quality ranking. No statement of
human alignment is supported without a separately documented annotation study.

## Result contract

Results use their own `schema_version: "1.0"`; do not rely on the manifest schema
version to describe output structure. The evaluation implementation must expose:

| Level | Required information |
| --- | --- |
| Run | Dataset ID; schema, package and contract versions; effective settings; input-manifest SHA-256; UTC start time; wall duration in seconds; submitted/evaluated/failed sample counts; sample records |
| Sample identity | Sample/group IDs; kind; instruction and provenance copied without invented values |
| Input identity | SHA-256 for each asset's original file bytes; decoded width/height; input modes and applied grayscale conversion |
| Support | Total, permitted-edit, preservation pixel counts; valid SSIM-window count if applicable |
| Measurements | Named per-metric value, status, support unit/count, parameters and direction |
| Diagnostics | Sample status, warnings, errors, and sample wall duration in seconds |

`evaluated` counts samples that loaded and reached measurement, including samples with
no applicable preservation metric. `failed` counts sample-level input/processing errors;
submitted equals evaluated plus failed on a completed run. Samples retain manifest order.
Run-level manifest errors prevent evaluation and must not produce a success summary.
Wall timing includes loading and measurement, excludes report rendering, and is not a
CPU-compute-time measurement. Package/dependency versions are reproducibility metadata;
absolute paths, user names, and private machine details do not belong in public reports.

Sample status is `ok` when measurements were reached (even if some are not applicable)
or `error` on sample failure. Warnings do not change `ok` to `error`. Metric status is
`ok`, `perfect_match` (PSNR only), or `not_applicable`. A measurement exception is a
sample error, not a fabricated numerical value. Invalid samples have no metric records;
retain their IDs and diagnostics and mark unavailable input metadata as null.

Warnings/errors carry a stable code and human-readable message. Initial codes include
`unknown_provenance`, `empty_preservation_region`, `insufficient_ssim_support`,
`missing_asset`, `invalid_image`, `invalid_mask`, `dimension_mismatch`,
`unsupported_input`, and `input_limit_exceeded`. Messages must avoid exposing absolute
paths in shareable output. Manifest structural errors use `invalid_manifest` at run level.

All serialized numbers must be finite. Undefined metric values are JSON null plus a
status/reason, never zero placeholders or nonstandard NaN/Infinity. Exact JSON nesting
and CSV column layout are outside this semantic contract.
Errors must not silently exclude difficult examples from counts or comparisons.

## Expected behavior by case

| Case | Expected behavior | Interpretation limit |
| --- | --- | --- |
| Original equals edit; preservation support exists | MAE 0; PSNR perfect-match status; SSIM 1 if supported | Instruction may still have failed |
| Changes confined to permitted edit region | Exact preservation for MAE/PSNR and supported SSIM windows | Does not establish correct edited content |
| Any changed channel outside permitted region | Positive MAE and finite PSNR; supported SSIM may change | Could reflect a mask/alignment problem, not necessarily a generator defect |
| All-zero mask | Measure the entire image | No permitted edit region |
| All-255 mask | Preservation metrics not applicable; warning; sample evaluated | Cannot claim preservation from no evidence |
| Missing mask or image | Sample error; keep record; no measurements | Do not infer a mask or drop the sample |
| Size mismatch | Sample error before measurements | No automatic resize or registration |
| Non-binary mask | Sample error | No implicit thresholding |
| Tiny preservation region | MAE/PSNR if pixels exist; SSIM may be not applicable | Distinguish metric support requirements |
| Valid dimensions but shifted content | Evaluate supplied pixels without correction | Pixel change is not semantic failure detection |
| Nonidentity orientation, unsupported color/alpha, or oversized input | Sample error before measurements | Upstream preprocessing must be explicit |

The specification does not define video/3D metrics, a human rating protocol, asset
generation, a leaderboard, or cross-modality quality comparisons.
