"""DGL versus ONNX user-facing latency in the same interpreter/environment."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser()
p.add_argument("--worker",choices=["dgl","onnx"])
p.add_argument("--case")
p.add_argument("--trials",type=int,default=10)
p.add_argument("--checkpoint",type=Path,required=True)
p.add_argument("--model",type=Path,default=ROOT/".artifacts/model")
a=p.parse_args()
if a.worker:
    start=time.perf_counter()
    if a.worker=="dgl":
        import torch
        import espaloma as esp
        torch.set_num_threads(8)
        torch.set_num_interop_threads(1)
    else:
        sys.path.insert(0,str(ROOT/"src"))
        from espaloma_onnx import Predictor
    imports_ms=(time.perf_counter()-start)*1000
    cases=json.loads((ROOT/"tests/reference/manifest.json").read_text())
    smiles=next(c["smiles"] for c in cases if c["name"]==a.case)
    start=time.perf_counter()
    if a.worker=="dgl":
        expected=json.loads((ROOT/"model-metadata/provenance.json").read_text())["upstream_checkpoint_sha256"]
        assert hashlib.sha256(a.checkpoint.read_bytes()).hexdigest()==expected
        model=torch.load(a.checkpoint,weights_only=False,map_location="cpu").eval()
        torch.set_grad_enabled(False)
        def predict():
            g=esp.Graph(smiles).heterograph
            model(g)
            # Materialize all corresponding output views, as in the ORT API.
            return [g.nodes[n].data[k] for n,k in (("n1","q"),("n1","s"),("n1","e"),("n2","k"),("n2","eq"),("n2","log_coefficients"),("n3","k"),("n3","eq"),("n3","log_coefficients"),("n4","k"),("n4_improper","k"))]
    else:
        predictor=Predictor.from_pretrained(a.model,threads=8)
        def predict():return predictor(smiles)
    load_ms=(time.perf_counter()-start)*1000
    start=time.perf_counter();predict();first_ms=(time.perf_counter()-start)*1000
    print(json.dumps(dict(imports_ms=imports_ms,load_ms=load_ms,first_request_ms=first_ms)),flush=True)
    for _ in range(10):predict()
    samples=[]
    for _ in range(100):
        start=time.perf_counter();predict();samples.append((time.perf_counter()-start)*1000)
    print(json.dumps(samples),flush=True)
else:
    import numpy as np
    def summary(v):return dict(median_ms=float(np.median(v)),p95_ms=float(np.percentile(v,95)))
    report=dict(threads=8,trials=a.trials,python=sys.version,scope="same Python interpreter and local files, 10 warmups/100 full SMILES requests per child; normal OS caches; fresh process time measured until first JSON response",results=[])
    for case in ("caffeine","cyclosporin"):
        for backend in ("dgl","onnx"):
            trials=[]
            for _ in range(a.trials):
                start=time.perf_counter()
                child=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),"--worker",backend,"--case",case,"--checkpoint",str(a.checkpoint),"--model",str(a.model)],stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,text=True)
                first=child.stdout.readline();ready=(time.perf_counter()-start)*1000
                if not first:raise RuntimeError("worker failed: "+case+" "+backend)
                row=json.loads(first);row["process_to_first_result_ms"]=ready
                row["warm_samples_ms"]=json.loads(child.stdout.readline())
                if child.wait():raise RuntimeError("worker failed")
                trials.append(row)
            metrics={k:summary([t[k] for t in trials]) for k in ("imports_ms","load_ms","first_request_ms","process_to_first_result_ms")}
            metrics["resident_request"]=summary([v for t in trials for v in t["warm_samples_ms"]])
            row=dict(molecule=case,backend=backend,summary=metrics,trials=trials)
            report["results"].append(row);print(case,backend,json.dumps(metrics),flush=True)
    (ROOT/"benchmarks/results/request-comparison.json").write_text(json.dumps(report,indent=2))
