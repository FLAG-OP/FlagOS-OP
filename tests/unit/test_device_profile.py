"""设备 profile 加载与模型路径覆盖（纯 CPU）。"""


def test_model_path_env_override(monkeypatch):
    from common.device import load_profile
    monkeypatch.setenv("FLAGOS_MODEL_PATH", "/tmp/my-model")
    p = load_profile("cpu")
    assert p.engine_args()["model"] == "/tmp/my-model"


def test_model_path_profile_default(monkeypatch):
    from common.device import load_profile
    monkeypatch.delenv("FLAGOS_MODEL_PATH", raising=False)
    p = load_profile("cpu")
    assert p.engine_args()["model"].endswith("llama-3.1-8b-like")
