from pathlib import Path
import hashlib
import json
import urllib.request
ROOT = Path(__file__).resolve().parents[1]
config = json.loads((ROOT / "model-metadata/config.json").read_text())
provenance = json.loads((ROOT / "model-metadata/provenance.json").read_text())
path = ROOT / ".benchmark/espaloma-0.3.2.pt"
path.parent.mkdir(exist_ok=True)
if not path.exists():
    urllib.request.urlretrieve(config["upstream_url"], path)
if hashlib.sha256(path.read_bytes()).hexdigest() != provenance["upstream_checkpoint_sha256"]:
    raise ValueError("Checkpoint checksum mismatch")
print(path)
