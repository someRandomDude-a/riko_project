"""Validated, content-addressed originals and metadata; no animation authoring."""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import struct
import threading

BONES = {'hips', 'spine', 'chest', 'upperChest', 'neck', 'head', 'leftEye', 'rightEye', 'jaw'}
for side in ('left', 'right'):
    BONES.update(side + name for name in ('UpperLeg', 'LowerLeg', 'Foot', 'Toes', 'Shoulder', 'UpperArm', 'LowerArm', 'Hand'))
    for finger in ('Thumb', 'Index', 'Middle', 'Ring', 'Little'):
        BONES.update(side + finger + joint for joint in ('Metacarpal', 'Proximal', 'Intermediate', 'Distal'))
STATES = {'idle', 'listening', 'thinking', 'speaking', 'tool', 'sleeping', 'held', 'clicked', 'settling', 'walking'}
MAX_BYTES = 32 * 1024 * 1024
DEFAULTS = {'enabled': True, 'julia_selection': True, 'policy_timeout_seconds': .75,
            'min_dwell_seconds': 1.0, 'transition_seconds': .3, 'min_confidence': .35,
            'mouse_tracking': True, 'walk_speed': 240.0}


def validate_settings(raw):
    if not isinstance(raw, dict): raise ValueError('animation must be a mapping')
    result = {**DEFAULTS, **raw}
    for key in ('enabled', 'julia_selection', 'mouse_tracking'):
        if type(result[key]) is not bool: raise ValueError(f'animation.{key} must be boolean')
    for key, low, high in [('policy_timeout_seconds', .05, 10), ('min_dwell_seconds', 0, 30),
                          ('transition_seconds', .05, 3), ('min_confidence', 0, 1), ('walk_speed', 20, 1000)]:
        value = result[key]
        if type(value) not in (int, float) or not math.isfinite(value) or not low <= value <= high:
            raise ValueError(f'animation.{key} must be between {low} and {high}')
    return result


def validate_pose(raw):
    if not isinstance(raw, dict) or raw.get('version') != 1: raise ValueError('Pose requires version 1')
    bones = raw.get('bones', {})
    expressions = raw.get('expressions', {})
    if not isinstance(bones, dict) or not bones or set(bones) - BONES: raise ValueError('Pose requires known humanoid bones')
    for rotation in bones.values():
        if not isinstance(rotation, list) or len(rotation) != 3 or any(type(v) not in (int, float) or not math.isfinite(v) or abs(v) > math.pi for v in rotation):
            raise ValueError('Pose rotations must be three finite Euler radians in [-pi, pi]')
    if not isinstance(expressions, dict) or len(expressions) > 64 or any(not isinstance(k, str) or not 1 <= len(k) <= 64 or type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 1 for k, v in expressions.items()):
        raise ValueError('Invalid pose expressions')
    return {'version': 1, 'bones': deepcopy(bones), 'expressions': deepcopy(expressions)}


def inspect_vrma(data):
    if len(data) < 20 or len(data) > MAX_BYTES: raise ValueError('Invalid VRMA size (maximum 32 MiB)')
    magic, version, length = struct.unpack_from('<4sII', data)
    if magic != b'glTF' or version != 2 or length != len(data): raise ValueError('VRMA must be a complete GLB v2 file')
    offset, document, binary_length = 12, None, 0
    while offset < len(data):
        if offset + 8 > len(data): raise ValueError('Truncated VRMA chunk')
        size, kind = struct.unpack_from('<II', data, offset)
        offset += 8
        if size % 4 or offset + size > len(data): raise ValueError('Invalid VRMA chunk length')
        if kind == 0x4E4F534A:
            if document is not None or offset != 20: raise ValueError('VRMA JSON must be the first chunk')
            try: document = json.loads(data[offset:offset + size].decode('utf-8').rstrip(' \x00'))
            except (UnicodeError, ValueError) as exc: raise ValueError('Invalid VRMA JSON') from exc
        elif kind == 0x004E4942: binary_length += size
        offset += size
    if not isinstance(document, dict): raise ValueError('VRMA has no JSON document')
    # External buffers/images would bypass the local asset boundary at render time.
    if any(item.get('uri') is not None for key in ('buffers', 'images') for item in document.get(key, [])):
        raise ValueError('VRMA must be self-contained; external/data URIs are not supported')
    if any(item.get('byteLength', 0) > binary_length for item in document.get('buffers', [])):
        raise ValueError('VRMA binary buffer is missing or truncated')
    extension = document.get('extensions', {}).get('VRMC_vrm_animation')
    if not isinstance(extension, dict) or not document.get('animations'): raise ValueError('File has no VRMC_vrm_animation animation')
    mapping = extension.get('humanoid', {}).get('humanBones', {})
    nodes = document.get('nodes', [])
    if not isinstance(mapping, dict) or set(mapping) - BONES: raise ValueError('Invalid VRMA humanoid bone mapping')
    for item in mapping.values():
        if not isinstance(item, dict) or type(item.get('node')) is not int or not 0 <= item['node'] < len(nodes):
            raise ValueError('VRMA bone references an invalid node')
    expressions = extension.get('expressions', {})
    names = list(expressions.get('preset', {})) + list(expressions.get('custom', {}))
    return {'bones': sorted(mapping), 'expressions': names[:64]}


def metadata(raw):
    if not isinstance(raw, dict): raise ValueError('Animation metadata must be an object')
    allowed = {'name', 'states', 'emotions', 'loop', 'layer', 'mask', 'transition_seconds', 'speed', 'license', 'source'}
    if set(raw) - allowed: raise ValueError('Unknown animation metadata field')
    result = {'name': 'Imported animation', 'states': [], 'emotions': [], 'loop': True,
              'layer': 'base', 'mask': [], 'transition_seconds': .3, 'speed': 1.0, 'license': '', 'source': '', **deepcopy(raw)}
    for key in ('name', 'license', 'source'):
        if not isinstance(result[key], str) or len(result[key]) > 1000: raise ValueError(f'Invalid animation {key}')
    for key, allowed_values in [('states', STATES), ('mask', BONES)]:
        value = result[key]
        if not isinstance(value, list) or len(value) > 100 or any(not isinstance(v, str) or v not in allowed_values for v in value): raise ValueError(f'Invalid animation {key}')
    if not isinstance(result['emotions'], list) or len(result['emotions']) > 32 or any(not isinstance(v, str) or not 1 <= len(v) <= 64 for v in result['emotions']): raise ValueError('Invalid animation emotions')
    if type(result['loop']) is not bool or result['layer'] not in {'base', 'gesture'}: raise ValueError('Invalid animation loop/layer')
    for key, low, high in [('transition_seconds', .05, 3), ('speed', .25, 3)]:
        value = result[key]
        if type(value) not in (int, float) or not math.isfinite(value) or not low <= value <= high: raise ValueError(f'Invalid animation {key}')
    return result


class AnimationLibrary:
    def __init__(self, root):
        self.directory = Path(root) / 'character_files' / 'animations'
        self.manifest = self.directory / 'library.json'
        self.lock = threading.RLock()
        self.entries = {}
        if self.manifest.exists():
            try:
                raw = json.loads(self.manifest.read_text(encoding='utf-8'))
                if raw['version'] != 1 or not isinstance(raw['entries'], list): raise ValueError('Invalid library schema')
                for entry in raw['entries']:
                    if entry['id'] in self.entries or not entry['id'].startswith('asset-'): raise ValueError('Invalid or duplicate animation ID')
                    entry = {**entry, **metadata({k: entry[k] for k in ('name', 'states', 'emotions', 'loop', 'layer', 'mask', 'transition_seconds', 'speed', 'license', 'source')})}
                    self._asset_path(entry)
                    self.entries[entry['id']] = entry
            except (ValueError, KeyError, TypeError) as exc: raise ValueError('Animation library is invalid; original manifest preserved') from exc

    def _asset_path(self, entry):
        path = (self.directory / entry['file']).resolve()
        if not path.is_relative_to(self.directory.resolve()) or path.suffix.lower() not in {'.vrma', '.json'}: raise ValueError('Invalid library asset path')
        return path

    def list(self):
        with self.lock: return deepcopy(list(self.entries.values()))

    def get(self, identifier):
        with self.lock:
            if identifier not in self.entries: raise ValueError('Unknown animation asset')
            return deepcopy(self.entries[identifier])

    def path(self, identifier):
        path = self._asset_path(self.get(identifier))
        if not path.is_file(): raise ValueError('Animation original is missing')
        return path

    def import_file(self, source, options=None):
        source = Path(source).expanduser().resolve()
        kind = 'vrma' if source.suffix.lower() == '.vrma' else 'pose' if source.name.lower().endswith('.pose.json') else None
        if kind is None: raise ValueError('Import supports .vrma and .pose.json; convert other formats first')
        if not source.is_file() or not 0 < source.stat().st_size <= MAX_BYTES: raise ValueError('Missing or oversized animation file')
        data = source.read_bytes()
        if len(data) > MAX_BYTES: raise ValueError('Animation file exceeds 32 MiB')
        if kind == 'vrma': details = inspect_vrma(data)
        else:
            try: pose = validate_pose(json.loads(data))
            except (UnicodeError, ValueError) as exc: raise ValueError(f'Invalid portable pose: {exc}') from exc
            details = {'bones': sorted(pose['bones']), 'expressions': sorted(pose['expressions']), 'pose': pose}
        digest = hashlib.sha256(data).hexdigest()
        identifier = 'asset-' + digest[:20]
        with self.lock:
            if identifier in self.entries: return self.get(identifier)
            entry = {'id': identifier, 'kind': kind, 'sha256': digest, 'file': f'originals/{digest}.{ "vrma" if kind == "vrma" else "pose.json"}',
                     **details, **metadata({'name': source.stem, **(options or {})})}
            path = self._asset_path(entry)
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists() and path.read_bytes() != data: raise ValueError('Existing animation original differs; preserved')
            if not path.exists(): path.write_bytes(data)
            self.entries[identifier] = entry
            try: self._save()
            except OSError:
                self.entries.pop(identifier)
                raise
            return deepcopy(entry)

    def update(self, identifier, options):
        with self.lock:
            previous = self.get(identifier)
            current = {k: previous[k] for k in ('name', 'states', 'emotions', 'loop', 'layer', 'mask', 'transition_seconds', 'speed', 'license', 'source')}
            updated = {**previous, **metadata({**current, **options})}
            self.entries[identifier] = updated
            try: self._save()
            except OSError:
                self.entries[identifier] = previous
                raise
            return deepcopy(updated)

    def _save(self):
        self.directory.mkdir(parents=True, exist_ok=True)
        temporary = self.manifest.with_suffix('.json.tmp')
        temporary.write_text(json.dumps({'version': 1, 'entries': list(self.entries.values())}, indent=2), encoding='utf-8')
        temporary.replace(self.manifest)
