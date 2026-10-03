"""Restrict renderer media to explicit local asset roots and inert media types."""
from pathlib import Path

EXTENSIONS = {'.png', '.jpg', '.jpeg', '.webp', '.gif', '.mp4', '.webm', '.mov', '.m4v'}


def resolve_media(root, value, effects_directory='effects/greenscreens', *, extensions=None):
    root = Path(root).resolve()
    roots = [root / 'character_files', root / 'persistent_memories' / 'generated_assets', root / effects_directory]
    path = Path(value)
    path = (path if path.is_absolute() else root / path).resolve()
    if not any(path.is_relative_to(directory.resolve()) for directory in roots): raise ValueError('Media path is outside approved asset directories')
    if path.suffix.lower() not in (EXTENSIONS if extensions is None else extensions): raise ValueError('Unsupported media type')
    if not path.is_file(): raise ValueError('Media asset does not exist')
    return path
