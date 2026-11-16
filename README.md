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
| DGL baseline | 8 | 23.61 ms | 26.86 ms |
| ONNX Runtime CPU | 8 | 11.81 ms | 15.75 ms |
| ONNX Runtime CPU | 16 | 12.50 ms | 13.41 ms |

This table measures model execution with prepared graph inputs. Full SMILES request and startup measurements are reported below. [Raw results](benchmarks/results/) include DGL, Tensor, compiled PyTorch and ONNX experiments; the [model card](docs/model-card.md) documents the model interface. Portable tests cover 116 original-DGL reference cases with exact input equality and output rtol=1e-4/atol=1e-5; 2 H2 cases fail original graph construction. Numerical consistency is not scientific force-field accuracy validation.

## First result and request latency

Ryzen 9 9950X3D, 8 CPU threads, local model files:

| User operation | Caffeine median / P95 | Cyclosporin A median / P95 |
|---|---:|---:|
| Start a Python process and receive the first result | 140.23 / 209.98 ms | 155.20 / 157.14 ms |
| Initialize Predictor (runtime import, file verification and session creation) | 47.87 / 56.10 ms | 47.02 / 48.41 ms |
| First SMILES request after Predictor initialization | 6.57 / 7.33 ms | 21.66 / 24.18 ms |
| Subsequent SMILES requests on a resident Predictor | 3.01 / 3.71 ms | 15.41 / 19.26 ms |

SMILES request timing covers molecular preprocessing, ONNX execution and result assembly. Process-to-first-result timing additionally covers Python startup, package imports and Predictor initialization. Each molecule was measured in 10 fresh subprocesses using local files with normal OS filesystem caching; each subprocess then performed 10 warmups and 100 requests (1,000 resident requests per molecule). Model download is a separate installation step. The rows measure different scopes; their medians are not additive.

Reproduce with `uv run python benchmarks/first_request.py --model /path/to/model --trials 10 --threads 8`.

Reuse a Predictor for interactive applications and services. For command-line usage, the process-to-first-result row represents the initial wait most directly. [Raw samples](benchmarks/results/first-request.json) include per-process startup stages and request timings.

[Export instructions](docs/export.md) · [Release automation](docs/publishing.md)

One molecule per call; dense O(N²) adjacency, original duplicated interaction conventions and native units. The package predicts parameters, not energies, forces or complete simulation systems. See the model card before downstream integration.
