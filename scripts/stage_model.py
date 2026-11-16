from pathlib import Path
import shutil
import hashlib
import json
ROOT = Path(__file__).resolve().parents[1]
out = ROOT / ".artifacts/model"
out.mkdir(parents=True, exist_ok=True)
if not (out / "model.onnx").is_file():
    raise FileNotFoundError("Export .artifacts/model/model.onnx first")
for path in (ROOT / "model-metadata").iterdir():
    shutil.copy(path, out / path.name)
shutil.copy(ROOT / "docs/model-card.md", out / "README.md")
for name in ("LICENSE", "NOTICE"):
    shutil.copy(ROOT / name, out / name)
shutil.copytree(ROOT / "benchmarks/results", out / "validation", dirs_exist_ok=True)
config = json.loads((out / "config.json").read_text())
config["sha256"] = hashlib.sha256((out / "model.onnx").read_bytes()).hexdigest()
(out / "config.json").write_text(json.dumps(config, indent=2))
provenance = json.loads((out / "provenance.json").read_text())
provenance["onnx_sha256"] = config["sha256"]
provenance["onnx_bytes"] = (out / "model.onnx").stat().st_size
(out / "provenance.json").write_text(json.dumps(provenance, indent=2))
(out / "SHA256SUMS").write_text("".join(hashlib.sha256(p.read_bytes()).hexdigest()+"  "+p.relative_to(out).as_posix()+"\n" for p in sorted(out.rglob("*")) if p.is_file() and p.name != "SHA256SUMS"))
