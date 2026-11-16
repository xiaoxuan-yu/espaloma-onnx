# Distribution and verification

Distribute the model together with its inference code, requirements, model card, LICENSE, NOTICE, provenance and validation reports. The model card and configuration specify the MIT license.

## Verify a downloaded repository

```bash
sha256sum -c SHA256SUMS
python -m pip install espaloma-onnx
espaloma-onnx CCO --model .
```

The runtime test checks SMILES inference, charge conservation, exact preprocessing inputs and numerical agreement with an original DGL reference fixture. See validation/ for expanded results and the model card for test scope.

The same files can be distributed as a custom model repository on Hugging Face or ModelScope. Repository distribution and hosted inference are separate integrations. `.gitattributes` specifies large-file storage for ONNX weights; check that downloads contain the model binary rather than an LFS pointer.

## Linux loader troubleshooting

If a conda ONNX Runtime binary fails to load with an executable-stack error, the tested environment used:

```bash
patchelf --clear-execstack <site-packages>/onnxruntime/capi/onnxruntime_pybind11_state.so
```

Apply this only for that specific loader error. Runtime dependency-isolation and clean uv wheel-install tests pass; see validation/package-validation.json.
