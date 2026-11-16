import hashlib
import json
import os
from pathlib import Path
import urllib.request
root = Path(__file__).resolve().parents[1]
headers = {"Accept":"application/vnd.github+json"}
if os.environ.get("GH_TOKEN"):
    headers["Authorization"] = "Bearer " + os.environ["GH_TOKEN"]
request = urllib.request.Request("https://api.github.com/repos/choderalab/espaloma/releases/latest", headers=headers)
release = json.load(urllib.request.urlopen(request, timeout=60))
config_path = root / "model-metadata/config.json"
config = json.loads(config_path.read_text())
version = release["tag_name"].lstrip("v")
changed = config["upstream_model"] != "espaloma-" + version
with open(os.environ["GITHUB_OUTPUT"], "a") as output:
    output.write("changed=" + str(changed).lower() + "\n")
if not changed:
    raise SystemExit(0)
assets = [a for a in release["assets"] if a["name"] == "espaloma-" + version + ".pt"]
if len(assets) != 1:
    raise ValueError("Expected exactly one versioned upstream .pt asset; manual review required")
data = urllib.request.urlopen(assets[0]["browser_download_url"], timeout=120).read()
config["upstream_model"] = "espaloma-" + version
config["upstream_url"] = assets[0]["browser_download_url"]
config_path.write_text(json.dumps(config, indent=2)+"\n")
path = root / "model-metadata/provenance.json"
p = json.loads(path.read_text())
p["upstream_checkpoint_sha256"] = hashlib.sha256(data).hexdigest()
p["upstream_release_url"] = release["html_url"]
path.write_text(json.dumps(p, indent=2)+"\n")
