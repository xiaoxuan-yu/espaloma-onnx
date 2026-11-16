import importlib.util
import importlib.metadata
import json
from pathlib import Path
import sys
import os
import types
import numpy as np
import pytest
from espaloma_onnx import Predictor, prepare
ROOT = Path(__file__).resolve().parents[1]
MODEL = Path(os.environ.get("ESPALOMA_ONNX_MODEL", ROOT / ".artifacts/model"))

@pytest.fixture(scope="module")
def predictor():
    return Predictor.from_pretrained(MODEL)

def test_minimal_environment():
    for name in ("torch", "dgl", "openff-toolkit", "espaloma"):
        with pytest.raises(importlib.metadata.PackageNotFoundError):
            importlib.metadata.distribution(name)

def test_reference_suite(predictor):
    fixtures = ROOT / "tests/reference"
    count = 0
    for case in json.loads((fixtures / "manifest.json").read_text()):
        if case["reference"] != "ok":
            continue
        ref = np.load(fixtures / (case["name"] + ".npz"))
        feed = prepare(case["smiles"])
        for key, value in feed.items():
            np.testing.assert_array_equal(value, ref[key])
        result = predictor(case["smiles"])
        for key, value in result.parameters.items():
            if "expected__" + key in ref:
                np.testing.assert_allclose(value, ref["expected__" + key], rtol=1e-4, atol=1e-5)
            else:
                assert value.shape[0] == 0
        np.testing.assert_allclose(result.parameters["n1_q"].sum(), result.total_charge, atol=1e-4)
        np.testing.assert_array_equal(result.indices["indices_2"], feed["indices_2"])
        count += 1
    assert count == 116

@pytest.mark.parametrize("source,module", [("huggingface","huggingface_hub"),("modelscope","modelscope")])
def test_download_adapter(monkeypatch, source, module):
    called = {}
    def download(*args, **kwargs):
        called.update(kwargs)
        return str(MODEL)
    monkeypatch.setitem(sys.modules, module, types.SimpleNamespace(snapshot_download=download))
    loaded = Predictor.from_pretrained("owner/test", source=source, revision="v1", local_files_only=True, cache_dir="cache")
    assert called["revision"] == "v1"
    assert called["local_files_only"] is True
    assert loaded("CCO").atom_features.shape == (9,114)

def test_invalid_model():
    with pytest.raises(ValueError):
        Predictor.from_pretrained("missing")
    with pytest.raises(ValueError):
        Predictor(MODEL, threads=0)

def test_invalid_smiles():
    with pytest.raises(ValueError):
        prepare("invalid_smiles")
