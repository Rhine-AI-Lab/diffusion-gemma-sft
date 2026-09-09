"""TRL/Unsloth training adapters for DiffusionGemma."""

# The RL trainers pull in trl's GRPOTrainer -> vLLM, which the SFT path does not
# need. Guard the import so SFT works in a vLLM-free environment.
try:
    from .trainer import DiffusionGemmaGDSDTrainer, DiffusionGemmaGRPOTrainer

    __all__ = ["DiffusionGemmaGDSDTrainer", "DiffusionGemmaGRPOTrainer"]
except Exception as _exc:  # noqa: BLE001 - optional RL deps (vllm, etc.)
    import warnings

    warnings.warn(f"RL trainers unavailable ({_exc!r}); SFT-only mode.")
    __all__ = []
