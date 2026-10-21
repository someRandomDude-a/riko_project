"""Rotating diagnostics; no chat text is collected by timing instrumentation. Every handler the backend installs redacts
secrets (SafeFormatter): debug.log, the console that basicConfig adds, and uvicorn's own (redact_handlers)."""
import logging
from logging.handlers import RotatingFileHandler
import re

FILE_FORMAT = '%(asctime)s %(levelname)-8s %(name)s: %(message)s'
CONSOLE_FORMAT = '%(asctime)s %(levelname)s %(name)s: %(message)s'
SECRET_NAMES = ('token', 'secret', 'password', 'api_key', 'confirm_key', 'private_key')  # environment and .env names whose values are redacted
PATTERN = re.compile(r'(?i)(bearer\s+|(?:api[_-]?key|token|password|secret)\s*[=:]\s*)[^\s,;]+')
REDACTED = set()  # values every SafeFormatter replaces: configure_logging's, then those minted later (redact_secrets: the API token)


def redact_secrets(*values):
    """Redact these values in every log handler from now on (8 characters or more: shorter ones would mangle ordinary text)."""
    REDACTED.update(str(value) for value in values if value and len(str(value)) >= 8)


class SafeFormatter(logging.Formatter):
    """fmt's format (or inner's: uvicorn's coloured ones), then every secret replaced: its own, REDACTED, and key=value
    or Bearer forms."""
    def __init__(self, secrets=(), fmt=FILE_FORMAT, inner=None):
        super().__init__(fmt)
        self.inner, self.secrets = inner, {str(value) for value in secrets if value and len(str(value)) >= 8}

    def format(self, record):
        text = self.inner.format(record) if self.inner is not None else super().format(record)
        for secret in sorted(self.secrets | REDACTED, key=len, reverse=True): text = text.replace(secret, '[REDACTED]')
        return PATTERN.sub(r'\1[REDACTED]', text)


def redact_handlers(*names):
    """Wrap the formatter of each handler on the named loggers in a SafeFormatter: uvicorn installs its own stderr handlers
    when its Config is built, after configure_logging."""
    for name in names:
        for handler in logging.getLogger(name).handlers:
            if not isinstance(handler.formatter, SafeFormatter): handler.setFormatter(SafeFormatter(inner=handler.formatter or logging.Formatter()))


def configure_logging(config):
    import os
    from dotenv import dotenv_values
    settings = config.raw.get('logging', {})
    level = str(settings.get('level', 'DEBUG' if config.raw.get('desktop', {}).get('debug') else 'INFO')).upper()
    if level not in {'DEBUG','INFO','WARNING','ERROR'}: level = 'INFO'
    # Every handler redacts these, whether or not debug.log is written: the console too (packaged, it is backend-launch.log).
    redact_secrets(*(value for key, value in {**dotenv_values(config.paths.env_file), **os.environ}.items() if any(word in key.lower() for word in SECRET_NAMES)),
        getattr(config.runtime, 'api_key', None))
    root = logging.getLogger()
    existing = list(root.handlers)
    logging.basicConfig(level=logging.INFO, format=CONSOLE_FORMAT)
    for handler in root.handlers:  # only the console basicConfig just added; a host's handlers (pytest's) are left alone
        if handler not in existing: handler.setFormatter(SafeFormatter(fmt=CONSOLE_FORMAT))
    for name in ('process.app_core','desktop_server'): logging.getLogger(name).setLevel(level)
    if not settings.get('inference_timings', True): logging.getLogger('process.app_core.kernel.metrics').setLevel(logging.WARNING)
    if settings.get('file_enabled', True):
        directory = config.paths.logs
        directory.mkdir(parents=True, exist_ok=True)
        handler = RotatingFileHandler(directory / 'debug.log', maxBytes=int(settings.get('max_mb',5))*1024*1024,
            backupCount=int(settings.get('backups',3)), encoding='utf-8')
        handler.setFormatter(SafeFormatter())
        handler.setLevel(level)
        logging.getLogger().addHandler(handler)
        logging.getLogger(__name__).info('Diagnostic logging enabled level=%s inference_timings=%s', level, settings.get('inference_timings',True))
