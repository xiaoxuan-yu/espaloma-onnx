import importlib.util
import sys
import pytest

def test_backend_import_is_lazy():
    import espaloma_onnx.openmm_backend
    assert "openmm" not in sys.modules

def test_optional_error():
    if importlib.util.find_spec("openmm") is not None:
        pytest.skip("Missing-extra message tested only in the base environment")
    from espaloma_onnx.openmm_backend import create_system
    with pytest.raises(ImportError,match="espaloma-onnx"):
        create_system({})
