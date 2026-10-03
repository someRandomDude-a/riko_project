"""Bounded attachment decoding with FFmpeg; never access a local microphone."""
import io
import subprocess

MAX_ATTACHMENT_BYTES = 8 * 1024 * 1024
MAX_PCM_BYTES = 16000 * 2 * 60
AUDIO_EXTENSIONS = frozenset({'.wav', '.mp3', '.ogg', '.opus', '.m4a', '.flac', '.webm', '.aac'})


def split_text(text, limit=1900):
    text = str(text) or '…'
    return [text[index:index + limit] for index in range(0, len(text), limit)]


def decode_audio(data, executable='ffmpeg'):
    if not data or len(data) > MAX_ATTACHMENT_BYTES: raise ValueError('Audio attachment must be nonempty and at most 8 MiB')
    command = [executable, '-nostdin', '-hide_banner', '-loglevel', 'error', '-max_alloc', '67108864',
        '-protocol_whitelist', 'pipe', '-i', 'pipe:0', '-t', '61', '-vn', '-ac', '1', '-ar', '16000', '-f', 's16le', 'pipe:1']
    try: result = subprocess.run(command, input=data, capture_output=True, timeout=30)
    except FileNotFoundError as exc: raise ValueError('FFmpeg is required for uploaded audio and Discord call playback') from exc
    except subprocess.TimeoutExpired as exc: raise ValueError('Audio decoding timed out') from exc
    if result.returncode or not result.stdout or len(result.stdout) % 2: raise ValueError('Could not decode this audio attachment')
    if len(result.stdout) > MAX_PCM_BYTES: raise ValueError('Audio must be at most 60 seconds')
    return result.stdout


def extract_pdf(data):
    from pypdf import PdfReader
    if len(data) > MAX_ATTACHMENT_BYTES: raise ValueError('PDF exceeds 8 MiB')
    document = PdfReader(io.BytesIO(data))
    if len(document.pages) > 50: raise ValueError('PDF exceeds 50 pages')
    text = '\n'.join((page.extract_text() or '')[:8000] for page in document.pages)
    return text[:12000] + ('\n[PDF excerpt truncated to 12000 characters]' if len(text) > 12000 else '')
