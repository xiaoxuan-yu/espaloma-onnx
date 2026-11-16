import importlib.abc
import sys
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split(".")[0] in {"torch","dgl","espaloma","openff"}:raise ImportError("Forbidden: "+fullname)
sys.meta_path.insert(0,Block())
import json
import time
from pathlib import Path
import numpy as np
from onnx_standalone import Predictor,prepare
root=Path(".benchmark/standalone-fixtures")
predictor=Predictor(".benchmark/espaloma-dynamic.onnx")
rows=[]
for case in json.loads((root/"manifest.json").read_text()):
 if case["reference"]!="ok":rows.append(case);continue
 row=dict(name=case["name"],status="error",stage="prepare")
 try:
  reference=np.load(root/(case["name"]+".npz"))
  start=time.perf_counter();feed=prepare(case["smiles"]);row["prepare_ms"]=(time.perf_counter()-start)*1000
  for key,value in feed.items():np.testing.assert_array_equal(value,reference[key],err_msg=key)
  row["stage"]="inference"
  start=time.perf_counter();actual=predictor(case["smiles"]);row["smiles_to_outputs_ms"]=(time.perf_counter()-start)*1000
  error=0.
  for key,value in actual.items():
   assert np.isfinite(value).all()
   if "expected__"+key not in reference:
    assert value.shape[0]==0;continue
   expected=reference["expected__"+key]
   np.testing.assert_allclose(value,expected,rtol=1e-4,atol=1e-5,err_msg=key)
   error=max(error,float(np.max(np.abs(value-expected))))
  np.testing.assert_allclose(actual["n1_q"].sum(),feed["charge"].sum(),atol=1e-4,rtol=1e-4)
  row.update(status="pass",max_absolute_error=error)
 except Exception as exc:row["error"]=str(exc)[:1500]
 rows.append(row)
 print(json.dumps(row),flush=True)
assert not {"torch","dgl","espaloma","openff"}.intersection(sys.modules)
report=dict(imports_blocked=True,results=rows,passed=sum(r.get("status")=="pass" for r in rows),failed=sum(r.get("status")=="error" for r in rows),reference_errors=sum(r.get("reference")=="error" for r in rows))
Path("benchmarks/standalone-validation.json").write_text(json.dumps(report,indent=2))
print(json.dumps({k:v for k,v in report.items() if k!="results"}))
if report["failed"]:raise SystemExit(1)
