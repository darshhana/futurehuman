import json, os, platform, random, time


def set_seed(seed=0):
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def snapshot(model=None, model_id=None, quant=None, seed=None, extra=None):
    """Everything the manual asks to record for reproducibility."""
    info = dict(timestamp=time.strftime("%Y-%m-%d %H:%M:%S"), python=platform.python_version(),
                platform=platform.platform(), model_id=model_id, quantization=quant, seed=seed)
    for name in ("torch", "transformers", "accelerate", "bitsandbytes", "peft", "datasets", "numpy", "pandas"):
        try:
            mod = __import__(name)
            info[name] = getattr(mod, "__version__", "?")
        except Exception:
            info[name] = None
    try:
        import torch
        info["cuda"] = torch.version.cuda
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            info["gpu_mem_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1024 ** 3, 2)
    except Exception:
        pass
    if model is not None:
        info["model_revision"] = getattr(getattr(model, "config", None), "_commit_hash", None)
    if extra:
        info.update(extra)
    return info


def save_snapshot(path, **kw):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    info = snapshot(**kw)
    with open(path, "w") as f:
        json.dump(info, f, indent=2)
    return info
