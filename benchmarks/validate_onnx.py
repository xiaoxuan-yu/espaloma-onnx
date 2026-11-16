"""Expanded conversion-equivalence suite; not force-field accuracy validation."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import onnxruntime as ort
import torch
import espaloma as esp
from rdkit import Chem
from tensor_model import TensorModel

CASES = {
"methane":"C", "ethane":"CC", "ethene":"C=C", "acetylene":"C#C", "water":"O", "ammonia":"N", "hydrogen":"[H][H]", "butane":"CCCC", "branched":"CC(C)(C)C", "long_chain":"C"*40,
"benzene":"c1ccccc1", "naphthalene":"c1ccc2ccccc2c1", "pyridine":"c1ccncc1", "pyrrole":"c1cc[nH]c1", "imidazole":"c1ncc[nH]1", "furan":"c1ccoc1", "thiophene":"c1ccsc1", "cyclopropane":"C1CC1", "cyclohexane":"C1CCCCC1", "bicyclic":"C1CC2CCC1C2", "macrocycle":"C1CCCCCCCCCCC1",
"ethanol":"CCO", "ether":"COC", "aldehyde":"CC=O", "ketone":"CC(=O)C", "acid":"CC(=O)O", "ester":"CC(=O)OC", "amide":"CC(=O)N", "urea":"NC(=O)N", "nitrile":"CC#N", "nitro":"C[N+](=O)[O-]", "guanidine":"NC(=N)N", "thiol":"CCS", "sulfide":"CSC", "sulfoxide":"CS(=O)C", "sulfone":"CS(=O)(=O)C", "sulfonamide":"CS(=O)(=O)N", "phosphate":"OP(=O)(O)O", "fluorinated":"FC(F)(F)C", "chlorinated":"ClCCCl", "brominated":"CCBr", "iodinated":"CCI",
"ammonium":"[NH4+]", "methylammonium":"C[NH3+]", "acetate":"CC(=O)[O-]", "guanidinium":"NC(N)=[NH2+]", "pyridinium":"c1cc[nH+]cc1", "zwitterion":"[NH3+]CC(=O)[O-]", "dication":"[NH3+]CC[NH3+]", "dianion":"[O-]C(=O)CC(=O)[O-]", "isotope":"[13CH3]CO", "chiral_R":"N[C@@H](C)C(=O)O", "chiral_S":"N[C@H](C)C(=O)O", "alkene_E":"C/C=C/C", "alkene_Z":"C/C=C\\C", "aspirin":"CC(=O)Oc1ccccc1C(=O)O", "caffeine":"CN1C=NC2=C1C(=O)N(C(=O)N2C)C", "peptide":"NCC(=O)N[C@@H](C)C(=O)NCC(=O)O"}
PROBES={"sodium":"[Na+]", "salt":"CC(=O)[O-].[Na+]", "disconnected":"CC.O", "radical":"[CH3]", "invalid":"not_a_smiles"}
p=argparse.ArgumentParser()
p.add_argument("--output",default="benchmarks/onnx-validation.json")
a=p.parse_args()
torch.set_num_threads(8)
torch.set_num_interop_threads(1)
model=torch.load(".benchmark/espaloma-0.3.2.pt",weights_only=False,map_location="cpu").eval()
options=ort.SessionOptions();options.intra_op_num_threads=8;options.inter_op_num_threads=1
options.add_session_config_entry("session.intra_op.allow_spinning","0")
path=Path(".benchmark/espaloma-dynamic.onnx")
session=ort.InferenceSession(str(path),sess_options=options,providers=["CPUExecutionProvider"])
input_names=[i.name for i in session.get_inputs()];output_names=[o.name for o in session.get_outputs()]
report={"onnx_sha256":hashlib.sha256(path.read_bytes()).hexdigest(),"ort":ort.__version__,"torch":torch.__version__,"rtol":1e-4,"atol":1e-5,"results":[]}
with torch.no_grad():
 for name,smi in {**CASES,**PROBES}.items():
  variants=[smi]
  mol=Chem.MolFromSmiles(smi)
  if name in CASES and mol is not None:
   variants.append(Chem.MolToSmiles(Chem.RenumberAtoms(mol,list(reversed(range(mol.GetNumAtoms())))),canonical=False,isomericSmiles=True))
  for variant,text in enumerate(variants):
   row=dict(name=name,variant=variant,smiles=text,probe=name in PROBES,stage="graph")
   try:
    graph=esp.Graph(text).heterograph
    row.update(atoms=graph.num_nodes("n1"),stage="reference")
    model(graph)
    row["stage"]="adapter"
    adapter=TensorModel(model,graph)
    feeds={n:getattr(adapter,n).numpy() for n in input_names}
    row["stage"]="onnx"
    actual=dict(zip(output_names,session.run(None,feeds)))
    row["stage"]="comparison"
    max_error=0.;checked=0
    for key,value in actual.items():
     node,feature=key.rsplit("_",1)
     if "log_coefficients" in key:node,feature=key.split("_log")[0],"log_coefficients"
     assert np.isfinite(value).all(),key+" non-finite"
     if graph.num_nodes(node)==0:
      assert value.shape[0]==0,key+" nonempty"
      continue
     expected=graph.nodes[node].data[feature].numpy()
     assert np.isfinite(expected).all(),key+" reference non-finite"
     np.testing.assert_allclose(value,expected,rtol=1e-4,atol=1e-5)
     checked+=1
     if value.size:max_error=max(max_error,float(np.abs(value-expected).max()))
    charge=float(adapter.charge.sum())
    np.testing.assert_allclose(actual["n1_q"].sum(),charge,atol=1e-4,rtol=1e-4)
    row.update(status="pass",max_absolute_error=max_error,checked_outputs=checked,total_charge=charge)
    if name == "aspirin" and variant == 0:
     np.savez(".benchmark/onnx-smoke-fixture.npz", **feeds,
              **{"expected__"+k:v for k,v in actual.items()})
   except Exception as exc:
    row.update(status="error",error=type(exc).__name__+": "+str(exc)[:2000])
   report["results"].append(row)
   print(json.dumps(row),flush=True)
report["summary"]={"passed":sum(r["status"]=="pass" for r in report["results"]),"errors":sum(r["status"]=="error" for r in report["results"]),"conversion_failures":sum(r["status"]=="error" and r["stage"] not in ("graph","reference") for r in report["results"])}
Path(a.output).write_text(json.dumps(report,indent=2))
print(json.dumps(report["summary"]))
if report["summary"]["conversion_failures"]:raise SystemExit(1)
