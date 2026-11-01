"""Pixel preservation measurements for validated, aligned LoadedSample inputs."""

import math

import numpy as np
from scipy.ndimage import binary_erosion
from skimage.metrics import structural_similarity

from visual_consistency.images import LoadedSample

SSIM_PARAMETERS = {
    "win_size": 7,
    "data_range": 1.0,
    "channel_axis": -1,
    "gaussian_weights": False,
    "use_sample_covariance": True,
    "K1": 0.01,
    "K2": 0.03,
}


def _metric(value, status, count, unit, direction, parameters):
    return {
        "value": value,
        "status": status,
        "support_count": count,
        "support_unit": unit,
        "direction": direction,
        "parameters": parameters.copy(),
    }


def measure_preservation(sample: LoadedSample) -> dict:
    """Measure loaded bytes without alignment, resizing, or instruction judgments."""
    original = sample.original.astype(np.float64) / 255.0
    edited = sample.edited.astype(np.float64) / 255.0
    difference = edited - original
    preservation = ~sample.edit_mask
    outside_count = int(preservation.sum())
    inside_count = int(sample.edit_mask.sum())
    parameters = {"data_range": 1.0, "channel_aggregation": "mean_rgb"}
    mae = mse = psnr = None
    status = "not_applicable"
    psnr_status = status
    if outside_count:
        outside = difference[preservation]
        mae = float(np.abs(outside).mean())
        mse = float(np.square(outside).mean())
        status = "ok"
        psnr_status = "perfect_match" if mse == 0 else "ok"
        psnr = None if mse == 0 else -10 * math.log10(mse)

    valid_windows = binary_erosion(
        preservation, structure=np.ones((7, 7), dtype=bool), border_value=0
    )
    window_count = int(valid_windows.sum())
    ssim = None
    diagnostics = []
    if window_count:
        _, similarity_map = structural_similarity(original, edited, full=True, **SSIM_PARAMETERS)
        ssim = float(similarity_map[valid_windows].mean())
    else:
        diagnostics.append(
            {
                "code": "insufficient_ssim_support",
                "message": "No complete 7x7 preservation windows are available.",
            }
        )

    inside_mae = float(np.abs(difference[sample.edit_mask]).mean()) if inside_count else None
    return {
        "support": {
            "total_pixels": int(preservation.size),
            "preservation_pixels": outside_count,
            "permitted_edit_pixels": inside_count,
            "ssim_windows": window_count,
        },
        "metrics": {
            "preservation_mae": _metric(mae, status, outside_count, "pixels", "lower", parameters),
            "preservation_psnr": _metric(
                psnr, psnr_status, outside_count, "pixels", "higher", parameters
            ),
            "preservation_ssim": _metric(
                ssim,
                "ok" if window_count else "not_applicable",
                window_count,
                "window_locations",
                "higher",
                SSIM_PARAMETERS,
            ),
            "inside_mask_mae": _metric(
                inside_mae,
                "ok" if inside_count else "not_applicable",
                inside_count,
                "pixels",
                "descriptive",
                parameters,
            ),
        },
        "warnings": diagnostics,
    }
