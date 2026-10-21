"""Restrict renderer media to explicit local asset roots and inert media types. A value (from the renderer, the model or a
config) is checked as text first (kernel/safe_paths.py), so a network path is refused before anything resolves it."""
import os
from pathlib import Path

from ..kernel.safe_paths import resolve_within, within

EXTENSIONS = {'.png', '.jpg', '.jpeg', '.webp', '.gif', '.mp4', '.webm', '.mov', '.m4v'}
EFFECTS_DIRECTORY = 'effects/greenscreens'  # desktop.effects_directory's default, against the data root


def effects_directory(raw):
    """desktop.effects_directory from a config's raw sections, relative to its root."""
    return raw.get('desktop', {}).get('effects_directory', EFFECTS_DIRECTORY)


def approved_roots(root, effects_directory=EFFECTS_DIRECTORY):
    """The three asset directories, each as given (under the data root) and as resolved: stored payload paths are resolved
    (/var is /private/var on macOS; a mapped drive resolves to its share on Windows; an effects directory may be a link),
    relative ones join the root as given. These come from the config, never from the value being checked."""
    given = [os.path.join(os.path.abspath(root), *parts) for parts in (('character_files',), ('persistent_memories', 'generated_assets'), (effects_directory,))]
    return list(dict.fromkeys([*given, *(str(Path(path).resolve()) for path in given)]))


def resolve_media(root, value, effects_directory=EFFECTS_DIRECTORY, *, extensions=None):
    allowed, roots, base = EXTENSIONS if extensions is None else extensions, approved_roots(root, effects_directory), os.path.abspath(root)
    if os.path.splitext(within(value, roots, base))[1].lower() not in allowed: raise ValueError('Unsupported media type')
    path = resolve_within(value, roots, base)  # only now is the filesystem consulted; a symlink pointing out is refused
    if path.suffix.lower() not in allowed: raise ValueError('Unsupported media type')
    if not path.is_file(): raise ValueError('Media asset does not exist')
    return path
