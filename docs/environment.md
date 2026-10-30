# Environment verification

Target: Windows, 64-bit Python 3.12, CPU execution only.

The dependency bounds are maintained in `pyproject.toml`; the exact tested environment
is recorded in `requirements/windows-cpu-py312.lock.txt`. The lock excludes the local
editable project and includes build tools needed for editable installation.

## Verified on October 30, 2026

- Windows 11 x64; Python 3.12.7.
- 16 logical CPU threads and approximately 15.7 GiB system RAM reported locally.
- PyTorch 2.15.0+cpu and torchvision 0.29.2+cpu; CUDA build absent.
- Editable package and ipykernel 7.4.0 import successfully.
- `python -m pip check`: no broken requirements.
- `python -m ruff check .`: passed.
- `python -m ruff format --check .`: passed.
- Five environment checks passed: scientific stack, PNG I/O, three-frame MJPG/AVI
  I/O, PLY mesh I/O, and CPU tensor execution.
- 66 exact dependency/build-tool versions captured in the environment lock.

The verification command `python -m pytest -p no:cacheprovider` passed all five
checks without warnings. The flag disables pytest's optional test-result cache.

Environment tests use tiny synthetic local fixtures; they establish package and file
I/O readiness, not research validity, model performance, real-dataset compatibility,
or 3D rendering support. The lock records installed versions, not artifact hashes;
reinstallation into a second clean environment has not been tested.
