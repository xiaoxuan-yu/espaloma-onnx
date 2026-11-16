"""Reference fixture producer; launches a dependency-blocked validation process."""
import ast
import json
import subprocess
import sys
from pathlib import Path
import numpy as np
import torch
import espaloma as esp
from tensor_model import TensorModel

root=Path(".benchmark/standalone-fixtures");root.mkdir(exist_ok=True)
source=ast.parse(Path("benchmarks/validate_onnx.py").read_text())
cases=next(eval(compile(ast.Expression(n.value), "cases", "eval"), {"__builtins__": {}}) for n in source.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=="CASES" for t in n.targets))
large=next(ast.literal_eval(n.value) for n in ast.parse(Path("benchmarks/cpu_large_ligand.py").read_text()).body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=="SMILES" for t in n.targets))
cases["cyclosporin"]=large
from rdkit import Chem
expanded={}
for name,smi in cases.items():
 expanded[name]=smi
 mol=Chem.MolFromSmiles(smi)
 if mol is not None:expanded[name+"_reordered"]=Chem.MolToSmiles(Chem.RenumberAtoms(mol,list(reversed(range(mol.GetNumAtoms())))),canonical=False,isomericSmiles=True)
torch.set_num_threads(8)
model=torch.load(".benchmark/espaloma-0.3.2.pt",weights_only=False,map_location="cpu").eval()
manifest=[]
with torch.no_grad():
 for name,smi in expanded.items():
  try:
   g=esp.Graph(smi).heterograph; adapter=TensorModel(model,g);model(g)
   data={n:getattr(adapter,n).numpy() for n in ("features","adjacency","indices_2","indices_3","indices_4","indices_4_improper","charge")}
   for key in adapter(adapter.features):
    node,feature=key.rsplit("_",1)
    if "log_coefficients" in key:node,feature=key.split("_log")[0],"log_coefficients"
    if g.num_nodes(node):data["expected__"+key]=g.nodes[node].data[feature].numpy()
   np.savez(root/(name+".npz"),**data)
   manifest.append(dict(name=name,smiles=smi,reference="ok"))
  except Exception as exc:manifest.append(dict(name=name,smiles=smi,reference="error",error=repr(exc)))
(root/"manifest.json").write_text(json.dumps(manifest,indent=2))
subprocess.run([sys.executable,"benchmarks/standalone_check.py"],check=True)
