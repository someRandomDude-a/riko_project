"""The allocation recommendation is configured demand, not a VRAM-fit ceiling."""
def suggested_pool(config):
    initiative, reflection = config.initiative_n_ctx, config.reflection_n_ctx
    background_slots = config.parallel_slots - 1
    background = max(reflection * background_slots, initiative + reflection * (background_slots - 1))
    return config.n_ctx + background if config.kv_unified else max(config.n_ctx, initiative, reflection) * config.parallel_slots


def pool_capacity(config):
    required = suggested_pool(config)
    if config.kv_pool_auto or config.kv_pool_tokens is None: return required
    if config.kv_pool_tokens < required: raise ValueError(f'KV pool must fit configured concurrent budgets ({required} tokens)')
    if not config.kv_unified and config.kv_pool_tokens % config.parallel_slots:
        raise ValueError('Separate KV pool size must be divisible by parallel slots')
    return config.kv_pool_tokens
