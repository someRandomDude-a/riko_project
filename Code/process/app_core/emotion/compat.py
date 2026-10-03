"""Compatibility with Julia's optional private-API encoder optimization."""
import logging


def compatible_engine(engine):
    encoder = getattr(getattr(engine, "model", None), "encoder", None)
    original = getattr(encoder, "_julia_original_forward", None)
    if original is not None and not hasattr(encoder, "_update_attention_mask"):
        # Preserve native Transformers masking rather than guessing mask semantics.
        encoder.forward = original
        engine.encoder_specialized = False
        logging.getLogger(__name__).info(
            "Julia: using standard Transformers encoder; private mask optimization unavailable"
        )
    return engine
