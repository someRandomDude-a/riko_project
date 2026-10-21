"""Leaf utilities every feature package may use: messages, stream filters, cancellation, the turn gate, runtime status and
turn context, workers, bounded cleanup, timings, torch device names, background budgets, the shared configuration checks
(validation.py), the settings schema every package registers its sections with (schema.py) and the typed voice, speech
and GPT-SoVITS sections beside the audio geometry (audio_config.py), and where the code and its workers are (code_paths.py).
kernel/ imports nothing else from app_core (tests/test_package_boundaries.py), and this file imports none of its modules, so
each loads on its own."""
