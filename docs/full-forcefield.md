# Complete force-field parameters with OpenFF

The optional `Predictor.create_forcefield` API returns backend-neutral parameter data. OpenFF assigns Lennard-Jones atom types and nonbonded rules; ONNX supplies partial charges and bonded parameters. Parameter assignment does not import OpenMM. A separate optional backend can construct an OpenMM System from the resulting data. The explicit default is `openff_unconstrained-2.1.1.offxml`, matching the upstream Espaloma template.

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

## Optional OpenMM System

Install OpenMM only when you need a simulation System:

```bash
uv pip install --python .pixi/envs/fullff/bin/python 'espaloma-onnx[openmm]>=0.2.0'
# For the development checkout instead:
uv pip install --python .pixi/envs/fullff/bin/python -e '.[openmm]'
```

```python
from openmm import XmlSerializer
ff = predict.create_forcefield('CCO')
system = ff.to_openmm_system()
with open('system.xml', 'w') as handle:
    handle.write(XmlSerializer.serialize(system))
```

OpenMM is an optional extra, not a base dependency. `to_openmm_system` constructs directly from the parameter dictionary and does not call OpenFF again. For saved JSON data, use `from espaloma_onnx.openmm_backend import create_system; system = create_system(data)`. This also works without installing OpenFF. Masses, harmonic terms, periodic terms, constraints, LJ and charge exceptions are included. It uses Lorentz–Berthelot combining rules and rejects unsupported alternatives. Positions and topology files are separate inputs to downstream simulation setup.

The default is explicit nonperiodic `NoCutoff`, appropriate for a single isolated molecule. `nonbonded_method='PME'` (or CutoffPeriodic/Ewald) requires `box_vectors_nm`; cutoff/switch settings are applied from the parameter metadata, with an optional `cutoff_nm` override. These simulation choices are explicit rather than inferred from the absence of a periodic box. The backend uses a standard NonbondedForce, optional CMMotionRemover and dispersion correction for periodic methods. PME/cutoff branches are configuration support; current numerical parity tests cover NoCutoff only.

Five NoCutoff molecular systems match original Espaloma deployment energies and forces at rtol=1e-4, atol=1e-3 (kJ/mol and kJ/mol/nm), with XML serialization round-trips. See benchmarks/results/openmm-backend.json. Base tests verify lazy loading and the missing-extra error, while the parameter-only smoke test blocks OpenMM imports.

The 0.2.0 wheel was installed with its OpenMM extra into a separate uv environment: OpenMM 8.6.1 wheel download was approximately 13.7 MiB. No OpenMM-CUDA packages, OpenFF, Torch or DGL were installed there. A saved ethanol parameter dictionary produced a 9-particle System, an XML round-trip and a CPU Reference Context. GPU plugins are separate OpenMM extras and are not selected by this package.
