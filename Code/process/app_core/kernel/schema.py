"""The settings schema: each package declares the YAML sections it reads, as a Section of Setting specs, and registers it
here, so configuration/schema.py derives load_config's checks, Settings' fields, the restart scopes and the live-apply
hooks without importing any feature package. kernel imports nothing else from app_core, so every package can declare its
own settings; app_core/__init__.py names the modules that declare them (loader()), which load the first time the
registry is read."""
from dataclasses import MISSING as NO_DEFAULT, dataclass, fields
import threading
from typing import Any, Callable

MISSING = type('Missing', (), {'__repr__': lambda self: 'MISSING', '__bool__': lambda self: False})()
# none: applies on save, through its live hook; python: restart the backend; electron: restart the desktop app, which reads
# it at launch (packaged, that restarts the backend too); microphone: applies the next time the microphone opens.
RESTART = ('none', 'python', 'electron', 'microphone')


@dataclass(frozen=True, slots=True)
class Setting:
    """One key of a section. What it leaves unset Settings infers: the kind from the value, whether it is a whole number
    from a numeric default (else from the value), and the group, UI section, restart scope, live hook, advanced and
    hidden flags from the sections that contain it."""
    key: str  # relative to its section, and may be dotted ('camera.fov')
    default: Any = MISSING  # what Settings lists while the YAML lacks the key; a callable takes the loaded AppConfig
    listed: bool = True  # False: Settings offers it only once the YAML has it
    kind: str | None = None  # boolean, number, text or json
    integer: bool | None = None
    nullable: bool = False  # null is a valid value: automatic, or not set
    options: tuple | None = None
    range: tuple | None = None  # (min, max) Settings accepts; load_config may accept a wider one
    restart: str | None = None  # RESTART
    live: str | None = None  # the hook that applies it on save (LIVE in app_core/http/settings.py)
    label: str | None = None
    help: str = ''
    group: str | None = None
    section: str | None = None  # the UI section it appears under
    advanced: bool | None = None
    readonly: bool = False
    hidden: bool | None = None  # still read (a legacy fallback), never offered
    obsolete: bool = False  # no longer read: never offered, and a strict section reports it as unknown
    quoted: bool = False  # always text, written quoted: PyYAML reads an unquoted 42 or yes as a number or boolean
    file: bool | None = None  # offer Browse and Check path
    visible_when: tuple = ()  # ((path, allowed values), ...): it matters only while each holds (provider-specific keys); the
    # renderer shows it only then (settings_model.mjs fieldRelevant), unless Advanced controls is on


@dataclass(frozen=True, slots=True)
class Rule:
    """A check across settings: Settings returns it to the UI (payload()) and its owner enforces it with broken(). It holds
    unless every `when` condition does and the requirement (at_most, below, one_of or any_set) fails."""
    path: str  # the setting the message is about
    message: str
    when: tuple = ()  # ((path, allowed values), ...)
    at_most: str = ''  # path's value must not exceed this setting's
    below: str = ''  # path's value must be smaller than this setting's
    one_of: tuple = ()  # ((path, allowed values), ...): each must hold
    any_set: tuple = ()  # groups of paths: every value of at least one group must be set (neither null nor empty)

    def broken(self, values):
        if not all(values.get(path) in allowed for path, allowed in self.when): return False
        value = values.get(self.path)
        for other, fits in ((self.at_most, lambda a, b: a <= b), (self.below, lambda a, b: a < b)):
            if other and value is not None and values.get(other) is not None and not fits(value, values[other]): return True
        if not all(values.get(path) in allowed for path, allowed in self.one_of): return True
        return bool(self.any_set) and not any(all(values.get(path) not in (None, '') for path in group) for group in self.any_set)

    def payload(self):
        found = {'path': self.path, 'message': self.message}
        for name in ('when', 'one_of'):
            if getattr(self, name): found[name] = {path: list(allowed) for path, allowed in getattr(self, name)}
        for name in ('at_most', 'below'):
            if getattr(self, name): found[name] = getattr(self, name)
        if self.any_set: found['any_set'] = [list(group) for group in self.any_set]
        return found


@dataclass(frozen=True, slots=True)
class Section:
    """A YAML mapping ('' is the top level) and the hooks configuration/schema.py runs for it. Several packages may add to
    one path (voice: kernel/audio_config.py and audio/asr.py); a check and each UI default come from one of them."""
    path: str
    settings: tuple = ()
    group: str | None = None  # the defaults for keys under path that no Setting of their own describes
    title: str | None = None  # the UI section
    restart: str | None = None
    live: str | None = None
    advanced: bool | None = None
    hidden: bool | None = None
    order: int = 0  # before path: Settings lists sections the renderer does not rank in this order
    strict: bool = False  # load_config reports keys that no Setting or `known` declares (kernel/validation.py unknown)
    known: frozenset = frozenset()  # other keys the section reads
    check: Callable | None = None  # check(mapping) -> its checked value (AppConfig.voice and the rest); raises naming the key
    configure: Callable | None = None  # configure(config, raw): checks across sections, once AppConfig is assembled
    effective: Callable | None = None  # effective(config, values): adjusts Settings' values after the YAML's
    review: Callable | None = None  # review(candidate, draft, changes) -> {path: error}: Settings' checks beyond load_config
    derive: Callable | None = None  # derive(candidate) -> {path: value}: values Settings writes with every save
    rules: tuple = ()

    def path_of(self, key): return f'{self.path}.{key}' if self.path else key

    def defaults(self):
        """{key: default} of its settings: the DEFAULTS of a section whose module keeps a plain mapping."""
        return {item.key: item.default for item in self.settings if item.default is not MISSING and not callable(item.default)}

    def contains(self, path): return not self.path or path == self.path or path.startswith(self.path + '.')


UNIQUE = ('check', 'group', 'title', 'restart', 'live', 'advanced', 'hidden')  # at most one section of a path sets each


class Registry:
    def __init__(self):
        self._sections, self._settings, self._loaders, self._ordered = [], {}, [], ()
        self._lock, self._loading, self._loaded, self._running = threading.Lock(), threading.RLock(), True, False

    def register(self, *sections):
        """Add sections (a module registers its own when it loads); returns the first. A setting declared twice, or a check
        or UI default two sections of one path both set, fails here instead of one silently replacing the other."""
        with self._lock:
            for section in sections:
                if any(section is known for known in self._sections): continue
                paths = [section.path_of(item.key) for item in section.settings]
                twice = sorted({path for path in paths if paths.count(path) > 1 or path in self._settings})
                if twice: raise ValueError(f'Settings declared twice: {", ".join(twice)}')
                for other in self._sections:
                    clash = [name for name in UNIQUE if other.path == section.path and getattr(other, name) is not None and getattr(section, name) is not None]
                    if clash: raise ValueError(f'Section {section.path!r} sets {", ".join(clash)} twice')
                self._sections.append(section)
                self._settings.update(zip(paths, section.settings))
            self._ordered = tuple(sorted(self._sections, key=lambda section: (section.order, section.path)))  # stable: a path's keep their order
        return sections[0] if sections else None

    def loader(self, function):
        """function() registers the sections of every package that declares any; it runs once, on first read."""
        with self._lock: self._loaders.append(function); self._loaded = False

    def _load(self):
        if self._loaded: return
        with self._loading:
            if self._loaded or self._running: return  # a loader that reads the registry gets what is there so far
            self._running = True
            try:
                for function in list(self._loaders): function()
                self._loaded = True
            finally: self._running = False

    def sections(self):
        """Every section, ordered by path, so checks and Settings' listing never depend on which module loaded first."""
        self._load()
        return self._ordered

    def setting(self, path):
        self._load()
        return self._settings.get(path)

    def resolve(self, path, name, title=False):
        """A setting's attribute, else the attribute of the innermost section containing path that sets it ('section'
        reads each Section's title)."""
        item = self.setting(path)
        if item is not None and getattr(item, name) is not None: return getattr(item, name)
        for section in sorted((s for s in self.sections() if s.contains(path)), key=lambda section: -len(section.path) if section.path else 1):
            value = getattr(section, 'title' if title else name)
            if value is not None: return value
        return None


REGISTRY = Registry()


def register(*sections): return REGISTRY.register(*sections)


def loader(function): return REGISTRY.loader(function)


def dataclass_defaults(cls):
    """{field: default} of a dataclass, without the fields built by a factory."""
    return {item.name: item.default for item in fields(cls) if item.default is not NO_DEFAULT}
