import argparse
import ast
import json
import time
from pathlib import Path
import numpy as np
import torch
import espaloma as esp
from tensor_model import TensorModel

p=argparse.ArgumentParser()
p.add_argument("--threads",type=int,default=8)
p.add_argument("--compile",action="store_true")
p.add_argument("--repeats",type=int,default=100)
a=p.parse_args()
torch.set_num_threads(a.threads)
torch.set_num_interop_threads(1)
torch.manual_seed(0)
model=torch.load(".benchmark/espaloma-0.3.2.pt",weights_only=False,map_location="cpu").eval()
smiles=next(ast.literal_eval(n.value) for n in ast.parse(Path("benchmarks/cpu_large_ligand.py").read_text()).body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=="SMILES" for t in n.targets))
cases={"caffeine":"CN1C=NC2=C1C(=O)N(C(=O)N2C)C", "cyclosporin":smiles, "butane":"CCCC", "ammonium":"[NH4+]"}
report=[]
with torch.no_grad():
    for name,smi in cases.items():
        g=esp.Graph(smi).heterograph
        t=time.perf_counter(); pure=TensorModel(model,g); prep=(time.perf_counter()-t)*1000
        model(g)
        expected={}
        for key in pure(pure.features):
            node,feature=key.rsplit("_",1)
            if "log_coefficients" in key: node=key.split("_log")[0]; feature="log_coefficients"
            if g.num_nodes(node):
                expected[key]=g.nodes[node].data[feature].clone()
        runners={"dgl":lambda:model(g),"tensor":lambda:pure(pure.features)}
        compile_ms=None
        if a.compile:
            t=time.perf_counter(); compiled=torch.compile(pure,fullgraph=True); compiled(pure.features); compile_ms=(time.perf_counter()-t)*1000
            runners["compiled"]=lambda:compiled(pure.features)
        timings={}
        errors={}
        for kind,run in runners.items():
            for _ in range(10): run()
            if kind!="dgl":
                actual=run()
                for key in expected:
                    torch.testing.assert_close(actual[key],expected[key],rtol=1e-4,atol=1e-5)
                errors[kind]=max((actual[k]-expected[k]).abs().max().item() for k in expected if expected[k].numel())
            samples=[]
            for _ in range(a.repeats):
                t=time.perf_counter();run();samples.append((time.perf_counter()-t)*1000)
            timings[kind]={"median_ms":float(np.median(samples)),"p95_ms":float(np.percentile(samples,95))}
        report.append(dict(molecule=name,atoms=g.num_nodes("n1"),threads=a.threads,preparation_ms=prep,compile_first_call_ms=compile_ms,checked_outputs=len(expected),max_absolute_errors=errors,timings=timings))
        print(json.dumps(report[-1]),flush=True)
Path(f"benchmarks/tensor-comparison-{a.threads}"+("-compiled" if a.compile else "")+".json").write_text(json.dumps(report,indent=2))
