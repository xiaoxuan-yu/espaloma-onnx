# Reproduce ONNX export

Run from the repository root. uv manages the runtime Python package; the root Pixi manifest locks the separate upstream OpenFF/DGL export environment.

```bash
pixi install --locked
mkdir -p .benchmark
curl -L --fail https://github.com/choderalab/espaloma/releases/download/0.3.2/espaloma-0.3.2.pt -o .benchmark/espaloma-0.3.2.pt
pixi run export
# Explicit paths, including from another working directory:
pixi run python scripts/export_model.py --checkpoint /path/to/espaloma-0.3.2.pt --output /path/to/model.onnx
ESPALOMA_ONNX_MODEL=.artifacts/model uv run pytest -q
```

The exporter verifies the original checkpoint hash from model-metadata/provenance.json, exports a dynamic single-molecule opset-17 graph, and runs the ONNX checker. Compare numerical outputs with tests/reference; binary hashes can vary with exporter versions. Export only trusted upstream checkpoints (torch.load unpickles code). Original checkpoint files are excluded from Git.

For the conda ONNX Runtime shared-library executable-stack issue on some newer Linux kernels, see docs/model-distribution.md. Runtime uv installations use the independent wheel dependencies. The output defaults to .artifacts/model/model.onnx. Neither generated model weights nor dist/ Python packages are committed. Run `uv run python scripts/stage_model.py` after export to assemble model metadata.

The `export` task first runs `prepare-export`, using the locked patchelf tool to clear the legacy executable-stack flag in libtorch_cpu.so before importing PyTorch. This environment-local binary compatibility step is idempotent and runs on every export. It changes neither checkpoint values nor ONNX computation. When invoking scripts/export_model.py directly, run `pixi run prepare-export` first.
