"""Normalize legacy CPU PyTorch executable-stack flags before importing it."""
from pathlib import Path
import subprocess
import sysconfig
library = Path(sysconfig.get_paths()["purelib"]) / "torch/lib/libtorch_cpu.so"
if not library.is_file():
    raise FileNotFoundError(library)
subprocess.run(["patchelf", "--clear-execstack", str(library)], check=True)
print("Prepared CPU export library:", library.name)
