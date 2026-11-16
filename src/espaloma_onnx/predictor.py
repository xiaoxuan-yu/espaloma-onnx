from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import numpy as np
from .preprocessing import prepare

@dataclass(frozen=True)
class Prediction:
    parameters: dict[str, np.ndarray]
    indices: dict[str, np.ndarray]
    atom_features: np.ndarray
    total_charge: float

class Predictor:
    def __init__(self, path, threads=8):
        import onnxruntime as ort
        if threads < 1:
            raise ValueError("threads must be positive")
        path = Path(path)
        if path.is_dir():
            path = path / "model.onnx"
        if not path.is_file():
            raise FileNotFoundError(path)
        config_path = path.with_name("config.json")
        if config_path.exists():
            config = json.loads(config_path.read_text())
            if config.get("upstream_model") != "espaloma-0.3.2" or config.get("format") != "onnx":
                raise ValueError("Unsupported model configuration")
            expected = config.get("sha256")
            if expected and hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise ValueError("Model checksum mismatch")
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        options.add_session_config_entry("session.intra_op.allow_spinning", "0")
        self.session = ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
        expected_inputs = {"features", "adjacency", "indices_2", "indices_3", "indices_4", "indices_4_improper", "charge"}
        if {i.name for i in self.session.get_inputs()} != expected_inputs:
            raise ValueError("Unsupported ONNX input schema")
        self.output_names = [o.name for o in self.session.get_outputs()]

    @classmethod
    def from_pretrained(cls, model, *, source=None, revision=None, cache_dir=None, local_files_only=False, threads=8):
        path = Path(model).expanduser()
        if path.exists():
            return cls(path, threads=threads)
        if source is None:
            raise ValueError('Local path does not exist; specify source="huggingface" or "modelscope" for a model ID')
        if source == "huggingface":
            try:
                from huggingface_hub import snapshot_download
            except ImportError as exc:
                raise ImportError("Install espaloma-onnx[hf] for Hugging Face downloads") from exc
            directory = snapshot_download(repo_id=str(model), revision=revision, cache_dir=cache_dir,
                                          local_files_only=local_files_only, allow_patterns=["model.onnx", "config.json"])
        elif source == "modelscope":
            try:
                from modelscope import snapshot_download
            except ImportError as exc:
                raise ImportError("Install espaloma-onnx[modelscope] for ModelScope downloads") from exc
            directory = snapshot_download(str(model), revision=revision, cache_dir=cache_dir,
                                          local_files_only=local_files_only, allow_patterns=["model.onnx", "config.json"])
        else:
            raise ValueError("source must be huggingface or modelscope")
        return cls(directory, threads=threads)

    def create_forcefield(self, smiles, *, forcefield="openff_unconstrained-2.1.1.offxml"):
        from .forcefield import create_forcefield
        return create_forcefield(self, smiles, forcefield=forcefield)

    def __call__(self, smiles):
        feed = prepare(smiles)
        parameters = dict(zip(self.output_names, self.session.run(None, feed)))
        return Prediction(parameters=parameters,
                          indices={key: value for key, value in feed.items() if key.startswith("indices_")},
                          atom_features=feed["features"], total_charge=float(feed["charge"].sum()))
