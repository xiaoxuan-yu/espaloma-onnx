"""Run complete parameter output with all simulation/training imports blocked."""
import importlib.abc
import json
from pathlib import Path
import sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"torch", "dgl", "espaloma", "openmm", "simtk"}:
            raise ModuleNotFoundError("Forbidden dependency: "+fullname, name=fullname)
sys.meta_path.insert(0,Block())
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
import numpy as np
from espaloma_onnx import Predictor
predictor=Predictor.from_pretrained(ROOT/".artifacts/model")
rows=[]
for smiles in ("CCO","CCCC","c1ccccc1","CC(=O)[O-]","CN1C=NC2=C1C(=O)N(C(=O)N2C)C"):
    result=predictor.create_forcefield(smiles);data=result.to_dict()
    assert np.isfinite(result.sigma_nm).all() and (result.sigma_nm>0).all()
    assert np.isfinite(result.epsilon_kj_mol).all() and (result.epsilon_kj_mol>=0).all()
    assert len(data["atoms"]["charge_e"])==len(result.sigma_nm)
    np.testing.assert_allclose(sum(data["atoms"]["charge_e"]),result.prediction.total_charge,atol=1e-4)
    assert json.loads(json.dumps(data))==data
    rows.append(dict(smiles=smiles,status="pass",atoms=len(result.sigma_nm)))
assert not {"torch","dgl","espaloma","openmm","simtk"}.intersection(sys.modules)
(ROOT/"benchmarks/results/full-forcefield-runtime.json").write_text(json.dumps(dict(imports_blocked=True,results=rows),indent=2))
print("PASS: 5 complete parameter outputs, JSON serialization, charge sums; OpenMM/training imports blocked")
