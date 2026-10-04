"""Rotating diagnostics; no chat text is collected by timing instrumentation."""
import logging
from logging.handlers import RotatingFileHandler
import re


class SafeFormatter(logging.Formatter):
    def __init__(self, secrets=()):
        super().__init__('%(asctime)s %(levelname)-8s %(name)s: %(message)s')
        self.secrets = sorted({str(value) for value in secrets if value and len(str(value)) >= 8}, key=len, reverse=True)

    def format(self, record):
        text = super().format(record)
        for secret in self.secrets: text = text.replace(secret, '[REDACTED]')
        return re.sub(r'(?i)(bearer\s+|(?:api[_-]?key|token|password|secret)\s*[=:]\s*)[^\s,;]+', r'\1[REDACTED]', text)


def configure_logging(config):
    import os
    from dotenv import dotenv_values
    settings = config.raw.get('logging', {})
    level = str(settings.get('level', 'DEBUG' if config.raw.get('desktop', {}).get('debug') else 'INFO')).upper()
    if level not in {'DEBUG','INFO','WARNING','ERROR'}: level = 'INFO'
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s: %(message)s')
    for name in ('process.app_core','desktop_server'): logging.getLogger(name).setLevel(level)
    if not settings.get('inference_timings', True): logging.getLogger('process.app_core.inference.metrics').setLevel(logging.WARNING)
    if settings.get('file_enabled', True):
        directory = config.root / 'logs'
        directory.mkdir(parents=True, exist_ok=True)
        secrets = [value for key,value in {**dotenv_values(config.root / '.env'), **os.environ}.items() if any(word in key.lower() for word in ('token','secret','password','api_key'))]
        secrets.append(getattr(config.runtime,'api_key',None))
        handler = RotatingFileHandler(directory / 'debug.log', maxBytes=int(settings.get('max_mb',5))*1024*1024,
            backupCount=int(settings.get('backups',3)), encoding='utf-8')
        handler.setFormatter(SafeFormatter(secrets))
        handler.setLevel(level)
        logging.getLogger().addHandler(handler)
        logging.getLogger(__name__).info('Diagnostic logging enabled level=%s inference_timings=%s', level, settings.get('inference_timings',True))
