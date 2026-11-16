"""Export once with dynamic graph inputs; test one ORT session on four molecules."""
import argparse
import ast
import json
import time
from pathlib import Path
import numpy as np
import onnx
import onnxruntime as ort
import torch
import espaloma as esp
from tensor_model import TensorModel

p = argparse.ArgumentParser()
p.add_argument("--threads", type=int, default=8)
p.add_argument("--repeats", type=int, default=100)
a = p.parse_args()
torch.set_num_threads(a.threads)
torch.set_num_interop_threads(1)
torch.manual_seed(0)
model = torch.load(".benchmark/espaloma-0.3.2.pt", weights_only=False, map_location="cpu").eval()
smiles = next(ast.literal_eval(n.value) for n in ast.parse(Path("benchmarks/cpu_large_ligand.py").read_text()).body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "SMILES" for t in n.targets))
cases = {"caffeine": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C", "cyclosporin": smiles, "butane": "CCCC", "ammonium": "[NH4+]"}
input_names = ["features", "adjacency", "indices_2", "indices_3", "indices_4", "indices_4_improper", "charge"]
def inputs(adapter):
    return tuple(getattr(adapter, n) for n in input_names)

def measure(run):
    for _ in range(10):
        run()
    samples = []
    for _ in range(a.repeats):
        start = time.perf_counter()
        run()
        samples.append((time.perf_counter() - start) * 1000)
    return {"median_ms": float(np.median(samples)), "p95_ms": float(np.percentile(samples, 95))}

with torch.no_grad():
    example = TensorModel(model, esp.Graph(cases["caffeine"]).heterograph)
    output_names = list(example(*inputs(example)))
    axes = {"features": {0: "atoms"}, "adjacency": {0: "atoms", 1: "atoms"}}
    for level in ("2", "3", "4", "4_improper"):
        axes["indices_" + level] = {0: "n" + level}
    for name in output_names:
        node = name.split("_log")[0] if "log_coefficients" in name else name.rsplit("_", 1)[0]
        axes[name] = {0: "atoms" if node == "n1" else node}
    path = Path(".benchmark/espaloma-dynamic.onnx")
    start = time.perf_counter()
    torch.onnx.export(example, inputs(example), str(path), input_names=input_names, output_names=output_names, dynamic_axes=axes, opset_version=17)
    onnx.checker.check_model(onnx.load(str(path)))
    export_ms = (time.perf_counter() - start) * 1000
    options = ort.SessionOptions()
    options.intra_op_num_threads = a.threads
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    # Avoid an idle ORT thread pool competing with the PyTorch baseline.
    options.add_session_config_entry("session.intra_op.allow_spinning", "0")
    start = time.perf_counter()
    session = ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
    session_ms = (time.perf_counter() - start) * 1000
    report = dict(threads=a.threads, repeats=a.repeats, warmup=10, onnx=onnx.__version__, onnxruntime=ort.__version__, torch=torch.__version__, model_bytes=path.stat().st_size, export_ms=export_ms, session_init_ms=session_ms, dynamic_model=True, spinning=False, results=[])
    for name, smi in cases.items():
        graph = esp.Graph(smi).heterograph
        start = time.perf_counter()
        adapter = TensorModel(model, graph)
        feed = {n: v.numpy() for n, v in zip(input_names, inputs(adapter))}
        preparation_ms = (time.perf_counter() - start) * 1000
        model(graph)
        start = time.perf_counter()
        actual = dict(zip(output_names, session.run(None, feed)))
        first_ms = (time.perf_counter() - start) * 1000
        max_error = 0.0
        checked = 0
        for key, value in actual.items():
            node, feature = key.rsplit("_", 1)
            if "log_coefficients" in key:
                node, feature = key.split("_log")[0], "log_coefficients"
            if not graph.num_nodes(node):
                assert value.shape[0] == 0
                continue
            expected = graph.nodes[node].data[feature].numpy()
            np.testing.assert_allclose(value, expected, rtol=1e-4, atol=1e-5)
            max_error = max(max_error, float(np.max(np.abs(value - expected))))
            checked += 1
        timings = {"dgl": measure(lambda: model(graph)), "tensor": measure(lambda: adapter(*inputs(adapter))), "onnxruntime": measure(lambda: session.run(None, feed))}
        row = dict(molecule=name, atoms=graph.num_nodes("n1"), preparation_ms=preparation_ms, first_ort_forward_ms=first_ms, checked_outputs=checked, max_absolute_error=max_error, timings=timings)
        report["results"].append(row)
        print(json.dumps(row), flush=True)
    Path(f"benchmarks/onnx-comparison-{a.threads}.json").write_text(json.dumps(report, indent=2))
