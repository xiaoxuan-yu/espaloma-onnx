---
license: mit
library_name: onnxruntime
tags:
- onnx
- chemistry
- molecular-mechanics
- cpu
- espaloma
language:
- en
---
# Espaloma 0.3.2 — ONNX CPU conversion

An FP32 ONNX conversion of the Espaloma 0.3.2 force-field parameter assignment model, with a CPU inference interface accepting SMILES. It predicts partial charges and bonded interaction parameters for downstream molecular mechanics workflows.

## Quick start

Download this model repository, then install the runtime package with Python 3.10:

```bash
pip install espaloma-onnx
espaloma-onnx CCO --model /path/to/model-repository
```

The package source and wheel build instructions are maintained at https://github.com/xiaoxuan-yu/espaloma-onnx . Runtime dependencies are RDKit, NumPy and ONNX Runtime.

```python
from espaloma_onnx import Predictor
predict = Predictor.from_pretrained('/path/to/model-repository', threads=8)
result = predict('CCO')
print(result.parameters['n1_q'])
print(result.indices['indices_2'], result.parameters['n2_eq'])
```

### Hugging Face

```bash
pip install 'espaloma-onnx[hf]'
```

```python
from espaloma_onnx import Predictor

predict = Predictor.from_pretrained(
    "youmans/espaloma-onnx",
    source="huggingface",
    revision="v0.1.0",
)
result = predict("CCO")
print(result.parameters["n1_q"])
```

### ModelScope

```bash
pip install 'espaloma-onnx[modelscope]'
```

```python
from espaloma_onnx import Predictor

predict = Predictor.from_pretrained(
    "youmans/espaloma-onnx",
    source="modelscope",
    revision="v0.1.0",
)
result = predict("CCO")
print(result.parameters["n1_q"])
```

The first load downloads and caches the model. Reuse the Predictor for subsequent requests. The GitHub source repository provides runtime code, build instructions and portable tests.

## Inputs and outputs

Each request processes one molecule, with dynamic atom and interaction counts. `prepare` adds explicit hydrogens, applies MDL aromaticity and generates seven arrays listed in `config.json`. Atom numbering follows RDKit parse order with hydrogens appended. Input feature definitions mirror upstream OpenFF/RDKit preprocessing, including removal of isotope information in the round-trip. Returned parameters follow the relation-index ordering; retain `prepare(smiles)` results for interpretation. Bonds, angles and proper torsions contain forward/reverse duplicates. Impropers follow the original checkpoint convention, including its permutations. Do not sum duplicate terms directly without the original Espaloma deployment conventions.

| Output | Shape | Meaning / native units |
|---|---|---|
| n1_s, n1_e | N x 1 | Learned charge-equilibration intermediates; do not interpret as standalone observables |
| n1_q | N x 1 | Partial charge, elementary charge |
| n2_log_coefficients | B x 2 | Internal bond parameterization |
| n2_k | B x 1 | Bond coefficient, Hartree/bohr² |
| n2_eq | B x 1 | Bond equilibrium distance, bohr |
| n3_log_coefficients | A x 2 | Internal angle parameterization |
| n3_k | A x 1 | Angle coefficient, Hartree/radian² |
| n3_eq | A x 1 | Angle equilibrium value, radians |
| n4_k | T x 6 | Proper torsion coefficients, Hartree, original periodic-term conventions |
| n4_improper_k | I x 2 | Improper coefficients, Hartree, original conventions |

Energy units follow upstream `espaloma.units`: Hartree per particle; distances are bohr and charges elementary charge. These are raw checkpoint outputs, not an OpenMM System. Consult original deployment code for multiplicities, periodicities, prefactors and unit conversion before constructing an energy expression. This conversion does not add missing nonbonded parameters or a complete force-field deployment pipeline.

## Validation

`validation/standalone-validation.json` records 118 tests: 116 pass and two H2 cases fail in original graph construction (empty angle-array shape bug). All seven standalone input arrays match the original path exactly for passing cases. ONNX outputs match DGL with rtol=1e-4, atol=1e-5; maximum absolute difference approximately 1.91e-6. Tests include 58 structures plus cyclosporin A, each with original/reversed atom-order SMILES. Standalone preprocessing and inference are also covered by runtime dependency-isolation tests.

The broader conversion report also includes boundary probes (isolated sodium, salts, radicals, invalid SMILES). Probe success means numerical consistency, not supported scientific use. These hand-selected tests are not comprehensive chemical-domain validation or energy/force accuracy validation. H2 has no reference equivalence result. The Python package wheel was also installed and tested in a separate uv environment containing only runtime dependencies; the CLI and original DGL aspirin reference checks passed.

## CPU performance

Ryzen 9 9950X3D, FP32, CPUExecutionProvider, 10 warmups and 100 sequential requests, one resident session:

| Molecule | Threads | ONNX median / P95 | Same-run DGL median / P95 |
|---|---:|---:|---:|
| Caffeine | 8 | 2.69 / 3.47 ms | 6.81 / 8.24 ms |
| Cyclosporin A (196 atoms with H) | 8 | 11.81 / 15.75 ms | 23.61 / 26.86 ms |
| Cyclosporin A | 16 | 12.50 / 13.41 ms | 21.15 / 25.32 ms |

The table measures ONNX execution using prepared NumPy graph inputs. Runtime settings are sequential execution, one inter-op thread, the indicated intra-op threads and disabled thread spinning. The following measurements cover the user-facing SMILES API and process startup.

## First result and request latency

Ryzen 9 9950X3D, 8 CPU threads, local model files:

| User operation | Caffeine median / P95 | Cyclosporin A median / P95 |
|---|---:|---:|
| Start a Python process and receive the first result | 140.23 / 209.98 ms | 155.20 / 157.14 ms |
| Initialize Predictor (runtime import, file verification and session creation) | 47.87 / 56.10 ms | 47.02 / 48.41 ms |
| First SMILES request after Predictor initialization | 6.57 / 7.33 ms | 21.66 / 24.18 ms |
| Subsequent SMILES requests on a resident Predictor | 3.01 / 3.71 ms | 15.41 / 19.26 ms |

SMILES request timing covers molecular preprocessing, ONNX execution and result assembly. Process-to-first-result timing additionally covers Python startup, package imports and Predictor initialization. Each molecule was measured in 10 fresh subprocesses using local files with normal OS filesystem caching; each subprocess then performed 10 warmups and 100 requests (1,000 resident requests per molecule). Model download is a separate installation step. The rows measure different scopes; their medians are not additive.

Reuse a Predictor for interactive applications and services. For command-line usage, the process-to-first-result row represents the initial wait most directly. [Raw samples](validation/first-request.json) include per-process startup stages and request timings.

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

Reproduce with an upstream export environment and its Python interpreter: `python benchmarks/compare_request.py --checkpoint /path/to/espaloma-0.3.2.pt --model /path/to/onnx-directory --trials 10`. The interpreter must provide upstream Espaloma and the runtime package source is resolved from this checkout. Run with the export environment activated so its AmberTools binaries are selected. [Full raw samples](validation/request-comparison.json).

## Limitations

- Dense adjacency requires O(N²) memory; intended for ligand-scale graphs, not proteins or huge systems.
- No batching, disconnected-fragment charge-partition validation, element-domain guarantee, energy/force validation or force-field assembly.
- Invalid SMILES and radicals are rejected; other unsupported chemistry may also raise errors. Feature vocabulary is not a scientific support list.
- Charges and atom indices must be preserved when integrating with downstream topology.
- This inference path does not expose gradients.

## Provenance, license and attribution

Upstream: https://github.com/choderalab/espaloma

Original checkpoint: https://github.com/choderalab/espaloma/releases/download/0.3.2/espaloma-0.3.2.pt

This ONNX conversion is distributed under the MIT License, following the upstream project license covering its officially released checkpoint. The original copyright and license text are included in [LICENSE](LICENSE), with attribution in [NOTICE](NOTICE). Redistribution must retain the copyright and permission notice; the software and model are provided without warranty. SHA256 checksums and source revision are in provenance.json and SHA256SUMS. Conversion source and export instructions are maintained in the GitHub source repository: https://github.com/xiaoxuan-yu/espaloma-onnx/blob/main/docs/export.md .

Please cite the original work: Wang, Fass and Chodera, “End-to-End Differentiable Construction of Molecular Mechanics Force Fields”, https://arxiv.org/abs/2010.01196 . This is an independently maintained conversion of the upstream model.

### Copyright

Copyright (c) 2026 Xiaoxuan Yu, Peking University. Original contributions to the conversion and inference package are licensed under MIT. Upstream Espaloma code and trained weights retain their original attribution. See LICENSE and NOTICE.
