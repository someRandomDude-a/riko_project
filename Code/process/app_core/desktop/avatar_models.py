"""Self-contained VRM library; imports copy bytes without overwriting user assets."""
import json
from pathlib import Path
import struct
import threading

MAX_BYTES = 128 * 1024 * 1024
FORMATS = ('auto', 'vrm0', 'vrm1')
LOCK = threading.RLock()


def inspect_model(path):
    path = Path(path)
    if path.suffix.lower() != '.vrm' or not path.is_file(): raise ValueError('Choose an existing .vrm file')
    if not 20 <= path.stat().st_size <= MAX_BYTES: raise ValueError('VRM size must be between 20 bytes and 128 MiB')
    with path.open('rb') as stream:
        header = stream.read(12)
        if len(header) != 12: raise ValueError('Truncated VRM header')
        magic, version, length = struct.unpack('<4sII', header)
        if magic != b'glTF' or version != 2 or length != path.stat().st_size: raise ValueError('VRM must be a complete GLB v2 file')
        header = stream.read(8)
        if len(header) != 8: raise ValueError('Truncated VRM JSON header')
        size, kind = struct.unpack('<II', header)
        if kind != 0x4E4F534A or size % 4 or size > 4 * 1024 * 1024 or size + 20 > length: raise ValueError('Invalid VRM JSON chunk')
        try: document = json.loads(stream.read(size).decode('utf-8').rstrip(' \x00'))
        except (UnicodeError, ValueError, RecursionError) as exc: raise ValueError('Invalid VRM JSON') from exc
        binary_length = 0
        while stream.tell() < length:
            header = stream.read(8)
            if len(header) != 8: raise ValueError('Truncated VRM chunk')
            chunk_size, chunk_kind = struct.unpack('<II', header)
            if chunk_size % 4 or stream.tell() + chunk_size > length: raise ValueError('Invalid VRM chunk length')
            if chunk_kind != 0x004E4942 or binary_length: raise ValueError('Unsupported VRM chunk')
            binary_length = chunk_size
            stream.seek(chunk_size, 1)
    if not isinstance(document, dict): raise ValueError('VRM JSON must be an object')
    for key in ('buffers', 'images'):
        items = document.get(key, [])
        if not isinstance(items, list) or any(not isinstance(item, dict) or 'uri' in item for item in items):
            raise ValueError('VRM must be self-contained; external/data URIs are not supported')
    buffers = document.get('buffers', [])
    if len(buffers) > 1 or any(type(item.get('byteLength')) is not int or not 0 <= item['byteLength'] <= binary_length for item in buffers):
        raise ValueError('Missing or invalid VRM binary buffer')
    extensions = document.get('extensions', {})
    if not isinstance(extensions, dict): raise ValueError('Invalid VRM extensions')
    if isinstance(extensions.get('VRMC_vrm'), dict) and extensions['VRMC_vrm'].get('specVersion') == '1.0': detected = 'vrm1'
    elif isinstance(extensions.get('VRM'), dict) and str(extensions['VRM'].get('specVersion', '')).startswith('0.'): detected = 'vrm0'
    else: raise ValueError('Model must contain a supported VRM 0.x or VRM 1.0 extension')
    extension = extensions['VRMC_vrm' if detected == 'vrm1' else 'VRM']
    humanoid = extension.get('humanoid', {})
    if not isinstance(humanoid, dict) or not humanoid.get('humanBones'): raise ValueError('VRM has no humanoid bone mapping')
    return detected


class AvatarModels:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.directory = self.root / 'character_files' / 'models'

    def listing(self):
        entries = []
        if not self.directory.resolve().is_relative_to(self.root): raise ValueError('Model directory must stay inside the repository')
        if self.directory.is_dir():
            for path in sorted(self.directory.iterdir(), key=lambda p: p.name.casefold()):
                resolved = path.resolve()
                if path.suffix.lower() == '.vrm' and resolved.is_relative_to(self.directory.resolve()) and resolved.is_file():
                    entries.append({'name': path.name, 'path': path.relative_to(self.root).as_posix()})
        return {'entries': entries, 'directory': 'character_files/models', 'formats': list(FORMATS)}

    def validate(self, value, model_format='auto'):
        if model_format not in FORMATS: raise ValueError('Choose Auto, VRM 0.x or VRM 1.0')
        path = Path(value).expanduser()
        path = (path if path.is_absolute() else self.root / path).resolve()
        if not path.is_relative_to((self.root / 'character_files').resolve()): raise ValueError('Import the VRM into character_files/models first')
        detected = inspect_model(path)
        if model_format != 'auto' and model_format != detected: raise ValueError(f'Model is {detected}; choose Auto or the matching format')
        return path, detected

    def import_model(self, value):
        source = Path(value).expanduser()
        source = (source if source.is_absolute() else self.root / source).resolve()
        detected = inspect_model(source)
        with LOCK:
            if not self.directory.resolve().is_relative_to(self.root): raise ValueError('Model directory must stay inside the repository')
            self.directory.mkdir(parents=True, exist_ok=True)
            if source.parent == self.directory.resolve():
                return {'path': source.relative_to(self.root).as_posix(), 'format': detected}
            # Exclusive creation preserves existing files, even when names collide.
            index = 0
            while True:
                destination = self.directory / (source.name if index == 0 else f'{source.stem}-{index}.vrm')
                try: output = destination.open('xb'); break
                except FileExistsError: index += 1
            try:
                with output, source.open('rb') as incoming:
                    total = 0
                    while chunk := incoming.read(1024 * 1024):
                        total += len(chunk)
                        if total > MAX_BYTES: raise ValueError('VRM exceeds 128 MiB')
                        output.write(chunk)
                detected = inspect_model(destination) # Reject a source changed during copying.
            except BaseException:
                destination.unlink(missing_ok=True)
                raise
        return {'path': destination.relative_to(self.root).as_posix(), 'format': detected}
