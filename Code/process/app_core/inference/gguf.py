"""GGUF metadata, read without loading the model: llama_native checks the training context, resources/ estimates VRAM."""
from pathlib import Path
import struct


def read_gguf(path):
    """Read only metadata, never tensors. Bound all untrusted lengths."""
    scalar = {0: '<B', 1: '<b', 2: '<H', 3: '<h', 4: '<I', 5: '<i', 6: '<f', 7: '<?', 10: '<Q', 11: '<q', 12: '<d'}
    with Path(path).open('rb') as file:
        def unpack(fmt):
            data = file.read(struct.calcsize(fmt))
            if len(data) != struct.calcsize(fmt): raise ValueError('Truncated GGUF metadata')
            return struct.unpack(fmt, data)[0]
        def string():
            length = unpack('<Q')
            if length > 4 * 1024 * 1024: raise ValueError('Oversized GGUF metadata string')
            data = file.read(length)
            if len(data) != length: raise ValueError('Truncated GGUF string')
            return data.decode('utf-8')
        def value(kind, keep=True, depth=0):
            if depth > 2: raise ValueError('Nested GGUF metadata exceeds limit')
            if kind in scalar: return unpack(scalar[kind])
            if kind == 8: return string()
            if kind == 9:
                subtype, count = unpack('<I'), unpack('<Q')
                if count > 2000000: raise ValueError('Oversized GGUF array')
                if subtype in scalar and not keep:
                    file.seek(struct.calcsize(scalar[subtype]) * count, 1)
                    return None
                result = []
                for _ in range(count):
                    item = value(subtype, False, depth + 1)
                    if keep and count <= 4096: result.append(item)
                return result if keep and count <= 4096 else None
            raise ValueError('Unknown GGUF metadata type')
        if file.read(4) != b'GGUF' or unpack('<I') not in (2, 3): raise ValueError('Not GGUF v2/v3')
        unpack('<Q')
        count = unpack('<Q')
        if count > 10000: raise ValueError('Oversized GGUF metadata')
        result = {}
        for _ in range(count):
            key, kind = string(), unpack('<I')
            keep = not key.startswith(('tokenizer.', 'general.description', 'general.tags'))
            item = value(kind, keep)
            if keep: result[key] = item
        return result
