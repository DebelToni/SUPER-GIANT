def register():
    # Lazy import string is recommended if CUDA init can happen at import time.
    # vLLM docs show both direct and lazy registration patterns.
    from vllm import ModelRegistry

    ModelRegistry.register_model(
        "ToyDecoderForCausalLM",
        "toy_vllm_model.model:ToyDecoderForCausalLM",
    )
