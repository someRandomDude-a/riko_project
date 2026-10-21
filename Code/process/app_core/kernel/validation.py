"""The checks every typed configuration section shares (kernel/audio_config.py, animation, wake feedback). Each names the
setting it rejects, so load_config stops with section.key and the allowed values instead of a turn or audio frame
failing later; a key no section declares is reported by unknown(), never a startup failure."""
import logging
import math
import operator

logger = logging.getLogger(__name__)
REPORTED = set()  # paths unknown() has logged: Settings runs load_config on every request


def section(path, raw):
    """The YAML mapping at path; an absent or emptied (null) section is {}."""
    if raw is None: return {}
    if not isinstance(raw, dict): raise ValueError(f'{path} must be a mapping of settings, not {raw!r}')
    return raw


def unknown(path, raw, known):
    """path.key for each key of the mapping raw that known does not declare, each logged once. A typo or a key an older
    version read keeps its default, and is named in the log and in Settings instead of stopping startup."""
    found = [f'{path}.{key}' for key in raw if key not in known] if isinstance(raw, dict) else []
    if not logging.getLogger().handlers: return found  # run_server loads the config before configuring logging; a later load logs
    for name in found:
        if name not in REPORTED:
            REPORTED.add(name)
            logger.warning('Ignoring %s: this version does not read it. Check its spelling, or remove it from character_config.yaml.', name)
    return found


def number(path, value, *, ge=None, gt=None, le=None, lt=None, integer=False):
    """value as a float (an int when integer) that is finite and within the given bounds. Numeric text is read too:
    PyYAML (YAML 1.1) reads 1e-3 or a quoted 1.5 as text, where ruamel and Electron (YAML 1.2) read a number."""
    parsed = value
    if isinstance(value, str) or type(value) is int and not integer:
        try: parsed = float(value)
        except ValueError: pass
        except OverflowError: parsed = math.inf  # an integer too large for a float
    bounds = [(limit, test, words) for limit, test, words in ((ge, operator.ge, 'at least'), (gt, operator.gt, 'above'),
        (le, operator.le, 'at most'), (lt, operator.lt, 'below')) if limit is not None]
    if (type(parsed) not in (int, float) or type(parsed) is float and not math.isfinite(parsed) or integer and parsed != int(parsed)
            or not all(test(parsed, limit) for limit, test, _ in bounds)):
        limits = ' and '.join(f'{words} {limit:g}' for limit, _, words in bounds)
        raise ValueError(f'{path} must be a {"whole" if integer else "finite"} number{" " + limits if limits else ""}, not {value!r}')
    return int(parsed) if integer else float(parsed)


def boolean(path, value):
    if type(value) is not bool: raise ValueError(f'{path} must be true or false, not {value!r}')
    return value


def choice(path, value, options):
    if value not in options: raise ValueError(f'{path} must be one of {", ".join(map(str, options))}, not {value!r}')
    return value


def text(path, value, *, optional=False):
    if value is None and optional: return None
    if not isinstance(value, str):
        raise ValueError(f'{path} must be text{" or null" if optional else ""}, not {value!r}: put numbers, yes, no, on and off in quotes')
    return value
