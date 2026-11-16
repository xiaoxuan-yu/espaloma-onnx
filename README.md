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

### Load from Hugging Face

```bash
uv sync --extra hf
```

```python
from espaloma_onnx import Predictor

predict = Predictor.from_pretrained(
    "youmans/espaloma-onnx",
    source="huggingface",
    revision="v0.1.0",
    threads=8,
)
result = predict("CCO")
print(result.parameters["n1_q"])
```

### Load from ModelScope

```bash
uv sync --extra modelscope
```

```python
from espaloma_onnx import Predictor

predict = Predictor.from_pretrained(
    "youmans/espaloma-onnx",
    source="modelscope",
    revision="v0.1.0",
    threads=8,
)
result = predict("CCO")
print(result.parameters["n1_q"])
```

Both downloaders cache model files locally. Set `cache_dir` to choose the cache location or `local_files_only=True` to use cached files offline. Retain the Predictor instance across requests.

## Complete force-field output

With the optional OpenFF base environment, `predict.create_forcefield("CCO")` returns charges, LJ, bonded parameters and nonbonded rules as backend-neutral data. `result.write_json("forcefield.json")` saves the complete parameter set. See [installation and validation](docs/full-forcefield.md). ONNX-only inference remains available through `predict(smiles)`.

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

## Full-request comparison: original DGL and ONNX

Same Ryzen 9 9950X3D, 8 CPU threads and Python interpreter, 10 fresh processes per molecule/backend. Each process performs one first request, 10 warmups and 100 subsequent requests. Local model files use normal OS caching. A request starts from SMILES and returns parameter arrays; DGL builds the original OpenFF/DGL graph on each request, while ONNX builds the seven RDKit/NumPy inputs. Model loading includes file checksum verification in both paths. Source model runtime is PyTorch 2.1.2/DGL 2.1.0+cu121 on CPU; ORT is 1.20.1. Preprocessing and framework imports are part of each API path.

| Molecule / operation | DGL median / P95 | ONNX median / P95 |
|---|---:|---:|
| Caffeine: process start to first result | 1298.83 / 1813.03 ms | 114.76 / 121.06 ms |
| Caffeine: first SMILES request after loading | 26.08 / 37.34 ms | 6.52 / 7.12 ms |
| Caffeine: subsequent SMILES request | 15.61 / 21.74 ms | 2.89 / 3.27 ms |
| Cyclosporin A: process start to first result | 1772.10 / 1819.01 ms | 128.38 / 132.49 ms |
| Cyclosporin A: first SMILES request after loading | 489.94 / 511.05 ms | 22.89 / 27.32 ms |
| Cyclosporin A: subsequent SMILES request | 391.11 / 486.25 ms | 15.22 / 19.94 ms |

For cyclosporin A, the full-request median is 25.7x faster on the independent ONNX path; process-to-first-result is 13.8x faster. This comparison measures the whole API, including the replacement of original graph preparation, rather than just the inference engine. Applications that cache original DGL graphs should use the prepared-input forward results above when estimating their benefit.

Reproduce with an upstream export environment and its Python interpreter: `python benchmarks/compare_request.py --checkpoint /path/to/espaloma-0.3.2.pt --model /path/to/onnx-directory --trials 10`. The interpreter must provide upstream Espaloma and the runtime package source is resolved from this checkout. Run with the export environment activated so its AmberTools binaries are selected. [Full raw samples](benchmarks/results/request-comparison.json).

[Export instructions](docs/export.md) · [Release automation](docs/publishing.md)

One molecule per call; dense O(N²) adjacency, original duplicated interaction conventions and native units. The package predicts parameters, not energies, forces or complete simulation systems. See the model card before downstream integration.
