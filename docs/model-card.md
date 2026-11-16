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

Download the repository and install the runtime dependencies with Python 3.10:

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
python inference.py 'CCO'
python examples/predict.py
python tests/test_runtime.py
```

The runtime uses RDKit for molecular preprocessing, NumPy for input arrays, and ONNX Runtime for CPU inference. Create one `Predictor` and reuse it across requests:

```python
from inference import Predictor, prepare
predict = Predictor('model.onnx', threads=8)  # retain this session
smiles = 'CCO'
feed = prepare(smiles)
parameters = predict(smiles)
print(parameters['n1_q'])
print(feed['indices_2'], parameters['n2_eq'])
```

## Python package

Install the package from PyPI after its release: `pip install espaloma-onnx`. Source and installation instructions: https://github.com/xiaoxuan-yu/espaloma-onnx . The repository-local inference.py remains available for direct use.

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

Forward timings exclude SMILES preprocessing, process startup and session initialization. Session initialization was approximately 14 ms. Results depend on hardware and system load. Runtime uses sequential execution, one inter-op thread, configurable intra-op threads, spinning disabled. See raw JSON reports.

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
