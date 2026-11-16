# espaloma-onnx

CPU inference for Espaloma 0.3.2: RDKit/NumPy preprocessing and a dynamic FP32 ONNX model. MIT licensed. Copyright (c) 2026 Xiaoxuan Yu, Peking University; upstream attribution is retained in LICENSE and NOTICE.

## Install and run

```bash
uv sync --locked
uv run espaloma-onnx CCO --model /path/to/model-directory
ESPALOMA_ONNX_MODEL=/path/to/model-directory uv run pytest -q
uv build
```

```python
from espaloma_onnx import Predictor
predict = Predictor.from_pretrained("/path/to/model-directory", threads=8)
result = predict("CCO")
print(result.parameters["n1_q"])
print(result.indices["indices_2"])
```

Optional download integrations: `uv sync --extra hf` or `uv sync --extra modelscope`; load `Predictor.from_pretrained("youmans/espaloma-onnx", source="huggingface", revision="version-tag")` (or source="modelscope"). Remote models must exist at the selected revision. Local paths work without either SDK.

## Layout

- src/espaloma_onnx/: installable runtime package and CLI
- tests/: portable original-DGL reference fixtures and regression tests
- model-metadata/: model configuration, license and upstream provenance
- docs/model-card.md: model card used for both model hubs
- .artifacts/: generated model assets (excluded from Git)
- benchmarks/: comparison tools and raw reports in results/
- scripts/: explicit-path export and platform publishing utilities
- docs/: export, comparison results and release setup
- pyproject.toml / uv.lock: runtime, development and optional download dependencies
- pixi.toml / pixi.lock: upstream export-only environment

## Comparison results

Ryzen 9 9950X3D, cyclosporin A (196 atoms including H), 10 warmups and 100 forwards:

| Runtime | Threads | Median | P95 |
|---|---:|---:|---:|
| DGL baseline (same ORT comparison run) | 8 | 23.61 ms | 26.86 ms |
| ONNX Runtime CPU | 8 | 11.81 ms | 15.75 ms |
| ONNX Runtime CPU | 16 | 12.50 ms | 13.41 ms |
| Pure Tensor + torch.compile (separate run) | 16 | 16.65 ms | 19.92 ms |

Forward timings exclude preprocessing/startup. Separate runs are not controlled head-to-head comparisons. [Raw results](benchmarks/results/) and [model card](docs/model-card.md) describe conditions and limitations. Portable tests cover 116 original-DGL reference cases with exact input equality and output rtol=1e-4/atol=1e-5; 2 H2 cases fail original graph construction. Numerical consistency is not scientific force-field accuracy validation.

[Export instructions](docs/export.md) · [Release automation](docs/publishing.md)

One molecule per call; dense O(N²) adjacency, original duplicated interaction conventions and native units. The package predicts parameters, not energies, forces or complete simulation systems. See the model card before downstream integration.
