from .base import *


_WORKER_IMPORTS = {
    "ChatGLMWorker": (".zhipu", "ChatGLMWorker"),
    "MiniMaxWorker": (".minimax", "MiniMaxWorker"),
    "XingHuoWorker": (".xinghuo", "XingHuoWorker"),
    "QianFanWorker": (".qianfan", "QianFanWorker"),
    "FangZhouWorker": (".fangzhou", "FangZhouWorker"),
    "QwenWorker": (".qwen", "QwenWorker"),
    "BaiChuanWorker": (".baichuan", "BaiChuanWorker"),
    "AzureWorker": (".azure", "AzureWorker"),
    "TianGongWorker": (".tiangong", "TianGongWorker"),
    "GeminiWorker": (".gemini", "GeminiWorker"),
}

_MODULE_IMPORTS = {
    "SparkApi": ".SparkApi",
}


def __getattr__(name):
    import importlib

    if name in _WORKER_IMPORTS:
        module_name, attr_name = _WORKER_IMPORTS[name]
        module = importlib.import_module(module_name, __name__)
        value = getattr(module, attr_name)
        globals()[name] = value
        return value

    if name in _MODULE_IMPORTS:
        module = importlib.import_module(_MODULE_IMPORTS[name], __name__)
        globals()[name] = module
        return module

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    name
    for name in globals()
    if not name.startswith("_") and name not in {"importlib"}
] + list(_WORKER_IMPORTS) + list(_MODULE_IMPORTS)
