"""Reproduce the small procedural fixtures; no generative model is involved."""

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent
SIZE = 256
BLUE = (45, 101, 190)
RED = (203, 65, 66)


def create_controls():
    rows, columns = np.indices((SIZE, SIZE))
    background = 238 - (rows // 24) - (columns // 32)
    original = np.stack((background, background + 3, background + 6), axis=-1).astype(np.uint8)
    original[180:184, 24:232] = (184, 191, 201)
    circle = (columns - 202) ** 2 + (rows - 57) ** 2 <= 16**2
    original[circle] = (218, 167, 55)
    original[88:168, 72:152] = BLUE

    mask = np.zeros((SIZE, SIZE), dtype=np.uint8)
    mask[88:168, 72:152] = 255
    correct = original.copy()
    correct[mask == 255] = RED

    outside = correct.copy()
    outside[36:60, 24:56] = (69, 139, 103)

    halo = correct.copy()
    band = np.zeros_like(mask, dtype=bool)
    band[84:172, 68:156] = True
    band[mask == 255] = False
    halo[band] = (232, 163, 165)

    shifted = np.clip(correct.astype(np.int16) + (8, -5, 3), 0, 255).astype(np.uint8)
    wrong = original.copy()
    wrong[mask == 255] = (62, 145, 104)

    cases = [
        ("identity", "Identity / no edit", original, mask, "Leave the image unchanged."),
        ("local-recolor", "Correct local recolor", correct, mask, None),
        ("outside-patch", "Unintended background patch", outside, mask, None),
        ("boundary-spill", "Color spills beyond mask", halo, mask, None),
        ("global-color-shift", "Global color shift", shifted, mask, None),
        ("unchanged-failed-edit", "Requested edit not performed", original, mask, None),
        ("wrong-local-color", "Wrong color inside mask", wrong, mask, None),
        (
            "no-preservation-support",
            "No preservation support",
            correct,
            np.full_like(mask, 255),
            None,
        ),
    ]
    assets = ROOT / "assets"
    assets.mkdir(exist_ok=True)
    Image.fromarray(original).save(assets / "original.png")
    Image.fromarray(mask).save(assets / "edit-mask.png")
    samples = []
    overview = Image.new("RGB", (768, 4 * 280), "white")
    draw = ImageDraw.Draw(overview)
    instruction = "Change the blue square to red while preserving its surroundings."
    for index, (sample_id, label, edited, case_mask, custom_instruction) in enumerate(cases):
        edited_path = f"assets/{sample_id}.png"
        mask_path = "assets/edit-mask.png"
        Image.fromarray(edited).save(ROOT / edited_path)
        if sample_id == "no-preservation-support":
            mask_path = "assets/full-edit-mask.png"
            Image.fromarray(case_mask).save(ROOT / mask_path)
        samples.append(
            {
                "sample_id": sample_id,
                "source_group_id": "geometric-scene-01",
                "original_path": "assets/original.png",
                "edited_path": edited_path,
                "mask_path": mask_path,
                "instruction": custom_instruction or instruction,
                "sample_kind": "synthetic_control",
                "provenance": {
                    "source": "Self-created geometric control; reproduced by create_controls.py.",
                    "source_url": None,
                    "license": "LicenseRef-Demo-Fixtures",
                    "license_reference": "LICENSE.txt",
                    "generator": None,
                },
            }
        )
        x, y = (index % 2) * 384, (index // 2) * 280
        overview.paste(Image.fromarray(edited), (x + 64, y + 24))
        draw.text((x + 12, y + 6), f"{index + 1}. {label}", fill=(28, 39, 56))
    overview.save(ROOT / "overview.png")
    manifest = {"schema_version": "1.0", "dataset_id": "image-controls-01", "samples": samples}
    (ROOT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    create_controls()
