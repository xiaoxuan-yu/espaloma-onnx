"""Single-request CPU latency, including separate preparation measurements."""
import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch
import espaloma as esp
from rdkit import Chem

# ChEBI:4031, https://www.ebi.ac.uk/chebi/CHEBI:91802
SMILES = "C/C=C/C[C@@H](C)[C@@H](O)[C@H]1C(=O)N[C@@H](CC)C(=O)N(C)CC(=O)N(C)[C@@H](CC(C)C)C(=O)N[C@@H](C(C)C)C(=O)N(C)[C@@H](CC(C)C)C(=O)N[C@@H](C)C(=O)N[C@H](C)C(=O)N(C)[C@@H](CC(C)C)C(=O)N(C)[C@@H](CC(C)C)C(=O)N(C)[C@@H](C(C)C)C(=O)N1C"
p = argparse.ArgumentParser()
p.add_argument("--threads", type=int, required=True)
p.add_argument("--repeats", type=int, default=200)
p.add_argument("--output", required=True)
a = p.parse_args()
torch.set_num_threads(a.threads)
torch.set_num_interop_threads(1)
torch.manual_seed(0)
start = time.perf_counter()
model = torch.load(".benchmark/espaloma-0.3.2.pt", map_location="cpu", weights_only=False).eval()
load_ms = (time.perf_counter() - start) * 1000
start = time.perf_counter()
graph = esp.Graph(SMILES).heterograph
graph_ms = (time.perf_counter() - start) * 1000
with torch.no_grad():
    start = time.perf_counter()
    model(graph)
    first_ms = (time.perf_counter() - start) * 1000
    start = time.perf_counter()
    for _ in range(10):
        model(graph)
    warmup_ms = (time.perf_counter() - start) * 1000
    samples = []
    start_all = time.perf_counter()
    for _ in range(a.repeats):
        start = time.perf_counter()
        model(graph)
        samples.append((time.perf_counter() - start) * 1000)
    total_ms = (time.perf_counter() - start_all) * 1000
finite = all(torch.isfinite(v).all().item() for n in graph.ntypes for _, v in graph.nodes[n].data.items() if v.is_floating_point())
assert finite, "non-finite graph tensor"
result = dict(molecule="cyclosporin A", smiles=SMILES, source="https://www.ebi.ac.uk/chebi/CHEBI:91802", heavy_atoms=Chem.MolFromSmiles(SMILES).GetNumHeavyAtoms(), atoms_including_hydrogens=graph.num_nodes("n1"), graph_nodes={n: graph.num_nodes(n) for n in graph.ntypes}, threads=a.threads, interop_threads=1, repeats=a.repeats, torch=torch.__version__, platform=platform.platform(), model_load_ms=load_ms, first_graph_build_ms=graph_ms, first_forward_ms=first_ms, additional_10_warmups_ms=warmup_ms, median_ms=float(np.median(samples)), p95_ms=float(np.percentile(samples,95)), p99_ms=float(np.percentile(samples,99)), mean_ms=float(np.mean(samples)), sequential_requests_per_second=a.repeats*1000/total_ms, finite=finite, scope="same prebuilt graph; forward only; no concurrency; model load and first graph build measured separately")
Path(a.output).write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
