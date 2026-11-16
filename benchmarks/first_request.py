"""Measure fresh-process time to first result and resident SMILES request latency."""
import argparse
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
p = argparse.ArgumentParser()
p.add_argument("--model", type=Path, default=ROOT / ".artifacts/model")
p.add_argument("--trials", type=int, default=10)
p.add_argument("--threads", type=int, default=8)
p.add_argument("--worker")
a = p.parse_args()
if a.worker:
    start = time.perf_counter()
    from espaloma_onnx import Predictor
    imports_ms = (time.perf_counter()-start)*1000
    manifest = json.loads((ROOT / "tests/reference/manifest.json").read_text())
    smiles = next(c["smiles"] for c in manifest if c["name"] == a.worker)
    start = time.perf_counter()
    predictor = Predictor.from_pretrained(a.model, threads=a.threads)
    load_ms = (time.perf_counter()-start)*1000
    start = time.perf_counter()
    result = predictor(smiles)
    first_ms = (time.perf_counter()-start)*1000
    print(json.dumps(dict(imports_ms=imports_ms, predictor_load_ms=load_ms, first_smiles_request_ms=first_ms, atoms=len(result.atom_features))), flush=True)
    # Warm measurements include complete SMILES preprocessing and output assembly.
    for _ in range(10): predictor(smiles)
    samples = []
    for _ in range(100):
        start = time.perf_counter(); predictor(smiles); samples.append((time.perf_counter()-start)*1000)
    print(json.dumps(dict(warm_samples_ms=samples)), flush=True)
else:
    report = dict(threads=a.threads, trials=a.trials, warmups=10, requests_per_trial=100,
                  scope="Fresh process, local model files; time-to-result ends when parent receives first JSON line; OS page caches are not flushed; no network downloads", results=[])
    for name in ("caffeine", "cyclosporin"):
        trials=[]
        for _ in range(a.trials):
            start=time.perf_counter()
            proc=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),"--worker",name,"--model",str(a.model),"--threads",str(a.threads)],stdout=subprocess.PIPE,text=True)
            first=proc.stdout.readline()
            ready_ms=(time.perf_counter()-start)*1000
            if not first: raise RuntimeError("Worker failed before first result")
            row=json.loads(first);row["process_to_first_result_ms"]=ready_ms
            row.update(json.loads(proc.stdout.readline()))
            if proc.wait()!=0:raise RuntimeError("Worker failed")
            trials.append(row)
        def summary(values):
            ordered=sorted(values)
            # Linear percentile interpolation, matching numpy default.
            k=(len(ordered)-1)*0.95;i=int(k)
            p95=ordered[i]+(ordered[min(i+1,len(ordered)-1)]-ordered[i])*(k-i)
            return dict(median_ms=statistics.median(values),p95_ms=p95)
        summaries={k:summary([t[k] for t in trials]) for k in ("imports_ms","predictor_load_ms","first_smiles_request_ms","process_to_first_result_ms")}
        summaries["resident_smiles_request"]=summary([v for t in trials for v in t["warm_samples_ms"]])
        report["results"].append(dict(molecule=name,atoms=trials[0]["atoms"],summary=summaries,trials=trials))
        print(name,json.dumps(summaries),flush=True)
    (ROOT / "benchmarks/results/first-request.json").write_text(json.dumps(report,indent=2))
