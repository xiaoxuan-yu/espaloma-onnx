# Complete force-field parameters with OpenFF

The optional `Predictor.create_forcefield` API returns backend-neutral parameter data. OpenFF assigns Lennard-Jones atom types and nonbonded rules; ONNX supplies partial charges and bonded parameters. The runtime does not generate simulation systems or import OpenMM. The explicit default is `openff_unconstrained-2.1.1.offxml`, matching the upstream Espaloma template.

## Install

uv manages the runtime package. Pixi supplies OpenFF Toolkit base, RDKit and OpenFF force-field data, which are distributed through conda-forge:

```bash
pixi install -e fullff --locked
uv pip install --python .pixi/envs/fullff/bin/python -e .
pixi run -e fullff python your_script.py
```

This environment contains neither OpenMM, CUDA, Torch, DGL, Interchange nor AmberTools. CPU ONNX Runtime is installed by uv. Installed environment size on the Linux test host is approximately 811 MB, including Python and all libraries (not merely the OpenFF namespace). `openff-toolkit-base` avoids the full toolkit metapackage's AmberTools dependency. Legacy OpenFF probes any antechamber executable on PATH; unrelated launcher shims may need removing from the shell PATH if they cause version-detection errors.

## API

```python
from espaloma_onnx import Predictor
predict = Predictor.from_pretrained('/path/to/model')
ff = predict.create_forcefield('CCO')
print(ff.sigma_nm, ff.epsilon_kj_mol)
ff.write_json('ethanol-forcefield.json')
parameters = ff.to_dict()
```

The JSON includes:

- Atom numbers, partial charges, LJ sigma/epsilon and matched OpenFF parameter IDs.
- Canonical bond/angle indices, equilibrium values and harmonic coefficients.
- Explicit proper/improper indices, periodicities, phases and energy coefficients.
- Any matched template constraints, including rigid-water distances.
- LJ combining rules, electrostatic settings, cutoff/switch settings, and graph-distance exclusion/1–4 scaling pairs.
- Units and functional forms. Lengths are nm, angles radians, energies kJ/mol, charges elementary charge.

Atom numbering is SMILES parse order followed by explicit hydrogens. Bond/angle/proper reverse duplicates are removed; improper conventions and their 0.5 factor follow original deployment. Negative torsion coefficients are converted to positive coefficients and shifted phases as in the upstream deployment. Exclusions use shortest bond-path distance; unlisted pairs have unit scaling. Predicting parameters alone uses `predict(smiles)` and does not import OpenFF. Coordinates and simulation backends are outside this API.

Only one connected molecule is accepted. Virtual sites and implicit-solvent handlers are rejected. Feature vocabulary is not a claim of broad chemical-domain support. Template selection is recorded in the result. Parameterization time includes OpenFF chemical typing and must be measured separately from ONNX-only latency.

## Validation

Five molecules (ethanol, butane, benzene, acetate and caffeine) were compared to the original deployment: charges, LJ particles, harmonic bonds/angles, proper/improper terms and nonbonded exceptions match at rtol=1e-4, atol=1e-5. OpenMM is used only in the original reference test environment, not by the package implementation. See benchmarks/results/full-forcefield.json.

A separate fullff runtime test blocks Torch/DGL/Espaloma/OpenMM imports and verifies five JSON outputs and total charge. The actual installed environment has no OpenMM or CUDA packages. See benchmarks/results/full-forcefield-runtime.json. These are parameter-equivalence tests, not scientific accuracy validation or energy/force claims for a new simulation implementation.
