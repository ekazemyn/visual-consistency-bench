"""Infrastructure checks only: no project metrics or evaluation logic."""

import cv2
import matplotlib
import numpy as np
import pandas as pd
import PIL.Image
import scipy
import sklearn
import torch
import torchvision
import trimesh
from skimage.metrics import structural_similarity


def test_scientific_stack():
    assert scipy.__version__
    assert sklearn.__version__
    assert matplotlib.__version__
    assert pd.DataFrame({"value": [1]}).shape == (1, 1)


def test_image_io(tmp_path):
    pixels = np.full((16, 16, 3), 127, dtype=np.uint8)
    path = tmp_path / "image.png"
    PIL.Image.fromarray(pixels).save(path)
    with PIL.Image.open(path) as image:
        loaded = np.asarray(image)
    np.testing.assert_array_equal(loaded, pixels)
    assert structural_similarity(pixels, loaded, channel_axis=2, data_range=255) == 1


def test_video_io(tmp_path):
    path = tmp_path / "clip.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 5, (32, 32))
    try:
        assert writer.isOpened(), "MJPG video writer is unavailable"
        for _ in range(3):
            writer.write(np.full((32, 32, 3), 127, dtype=np.uint8))
    finally:
        writer.release()
    capture = cv2.VideoCapture(str(path))
    try:
        assert capture.isOpened()
        frames = 0
        while True:
            success, frame = capture.read()
            if not success:
                break
            assert frame.shape == (32, 32, 3)
            frames += 1
        assert frames == 3
    finally:
        capture.release()


def test_mesh_io(tmp_path):
    path = tmp_path / "box.ply"
    trimesh.creation.box().export(path)
    mesh = trimesh.load_mesh(path)
    assert mesh.is_watertight
    assert len(mesh.faces) == 12


def test_cpu_torch():
    assert torch.version.cuda is None, "Install the CPU wheel using the setup guide"
    tensor = torch.ones((2, 2), device="cpu")
    assert (tensor @ tensor).sum().item() == 8
    assert torchvision.__version__
