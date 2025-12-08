def AutoModelForCausalLM(config):
    if config.model_type == "qwen2":
        from .qwen2 import Qwen2ForCausalLM

        config.attention_bias = True
        if getattr(config, "head_dim", None) is None:
            config.head_dim = config.hidden_size // config.num_attention_heads

        return Qwen2ForCausalLM(config)
    elif config.model_type == "qwen3":
        from .qwen3 import Qwen3ForCausalLM

        return Qwen3ForCausalLM(config)
    elif config.model_type == "llama":
        from .llama import LlamaForCausalLM

        return LlamaForCausalLM(config)
    else:
        raise ValueError(f"Unsupported method: {config.method}")

__all__ = ["AutoModelForCausalLM"]