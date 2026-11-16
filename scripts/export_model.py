"""Export the upstream checkpoint using explicit, cwd-independent paths."""
import argparse
import hashlib
from pathlib import Path
import sys
import onnx
import torch
import espaloma as esp
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks"))
from tensor_model import TensorModel
p = argparse.ArgumentParser()
p.add_argument("--checkpoint", type=Path, default=ROOT / ".benchmark/espaloma-0.3.2.pt")
p.add_argument("--output", type=Path, default=ROOT / ".artifacts/model/model.onnx")
a = p.parse_args()
expected = __import__("json").loads((ROOT / "model-metadata/provenance.json").read_text())["upstream_checkpoint_sha256"]
if hashlib.sha256(a.checkpoint.read_bytes()).hexdigest() != expected:
    raise ValueError("Upstream checkpoint checksum mismatch")
torch.set_num_threads(8)
model = torch.load(a.checkpoint, weights_only=False, map_location="cpu").eval()
graph = esp.Graph("CN1C=NC2=C1C(=O)N(C(=O)N2C)C").heterograph
adapter = TensorModel(model, graph)
names = ["features", "adjacency", "indices_2", "indices_3", "indices_4", "indices_4_improper", "charge"]
args = tuple(getattr(adapter, n) for n in names)
with torch.no_grad():
    outputs = list(adapter(*args))
axes = {"features": {0:"atoms"}, "adjacency": {0:"atoms",1:"atoms"}}
for level in ("2", "3", "4", "4_improper"):
    axes["indices_" + level] = {0:"n"+level}
for name in outputs:
    node = name.split("_log")[0] if "log_coefficients" in name else name.rsplit("_",1)[0]
    axes[name] = {0:"atoms" if node == "n1" else node}
a.output.parent.mkdir(parents=True, exist_ok=True)
with torch.no_grad():
    torch.onnx.export(adapter, args, str(a.output), input_names=names, output_names=outputs, dynamic_axes=axes, opset_version=17)
onnx.checker.check_model(onnx.load(str(a.output)))
print(a.output, hashlib.sha256(a.output.read_bytes()).hexdigest())
