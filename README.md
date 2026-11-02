# Evaluating Visual Consistency in AI-Generated Images, Videos, and 3D Assets

A CPU-first research project investigating how inexpensive consistency measurements
agree with human judgments, and where those measurements fail.

The research scope covers image-edit preservation, temporal consistency in video,
and consistency across views of generated 3D assets. Each modality requires its own
validated measurements.

## Setup

Use 64-bit Python 3.12. The verified environment targets Windows and CPU execution.
Run these PowerShell commands from the project root; activation is unnecessary:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip setuptools
.\.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m ruff check .
```

`requirements/windows-cpu-py312.lock.txt` records the exact tested dependencies.
To reproduce that environment instead:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements/windows-cpu-py312.lock.txt
.\.venv\Scripts\python.exe -m pip install --no-deps --no-build-isolation -e ".[dev]"
```

The lock is platform-specific. The CPU index supplies the pinned `+cpu` PyTorch
wheels; normal scientific packages come from PyPI. No model weights are downloaded.

In VS Code, select `.venv/Scripts/python.exe` as the Python interpreter and the
notebook kernel. The Python and Jupyter extensions are needed for notebook editing;
no separate Jupyter server is required.

## Evaluate image preservation

After setup, run the included eight synthetic controls:

```powershell
.\.venv\Scripts\python.exe -m visual_consistency examples/images/manifest.json outputs/image-demo
```

This writes `results.json` and `metrics.csv`. Existing results require `--overwrite`.
The installed `visual-consistency` command accepts the same arguments. Exit codes
are 0 for a completed run without sample failures, 1 for retained sample failures,
and 2 for invalid arguments, manifest errors, or output errors.

See the [structured output guide](docs/image-evaluation.md#command-line-results)
for result fields, support/status interpretation, and privacy limits. The controls
demonstrate pixel-change measurements; they do not validate generated-image quality.

## Stack

| Purpose | Tools |
| --- | --- |
| Arrays, statistics, tabular experiment results | NumPy, SciPy, pandas, scikit-learn |
| Image I/O and classical measurements | Pillow, scikit-image |
| Video decoding and frame operations | OpenCV headless |
| Mesh loading and geometry inspection | Trimesh |
| Deep learning | CPU PyTorch and torchvision |
| Research plots | Matplotlib |
| Environment verification and code quality | pytest, Ruff |
| Optional IDE notebook execution | ipykernel, installed with the dev extra |

OpenCV headless supports file operations without GUI windows. Trimesh is installed
without its rendering extras; the environment supports mesh I/O and geometry inspection.
See the official [OpenCV package guidance](https://pypi.org/project/opencv-python-headless/),
[Trimesh installation guide](https://trimesh.org/install), and
[PyTorch installation guide](https://pytorch.org/get-started/locally/).

## Layout

```text
src/visual_consistency/   Python package
tests/                   Dependency and local file I/O smoke checks
docs/                    Environment documentation
data/raw/                Local source assets (ignored)
data/processed/          Local derived assets (ignored)
models/                  Local model weights (ignored)
notebooks/               Exploratory notebooks
outputs/                 Generated results and figures (ignored)
requirements/            Tested environment snapshot
```

No cloud account, API key, CUDA installation, Docker, Blender, or external FFmpeg
executable is required for the environment checks. Video I/O is verified with
MJPG/AVI; support for other codecs depends on the installed OpenCV build.

See [environment verification](docs/environment.md) for tested capabilities and limitations.
The [image evaluation contract](docs/image-evaluation.md) defines input requirements,
mask semantics, and preservation-result interpretation.
The [image control collection](examples/images/README.md) includes eight reproducible
synthetic examples with masks, provenance, and documented expected behavior.
The [Python loading API](docs/image-evaluation.md#load-and-validate-inputs) validates
manifests and loads aligned RGB images and binary masks without resizing or scoring them.
The [measurement API](docs/image-evaluation.md#python-measurement-api) computes
preservation MAE, PSNR, and SSIM over fully supported windows, plus descriptive
inside-mask MAE. These are pixel-change diagnostics, not editing-success judgments.
