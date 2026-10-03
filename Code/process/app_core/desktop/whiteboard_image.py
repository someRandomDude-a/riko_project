"""Changed-only whiteboard exports; capture the board, never the user's desktop."""
import hashlib
import io
import json
import threading
import textwrap

from .media import resolve_media


def board_revision(snapshot):
    content = {'page': snapshot.get('whiteboard_page', 'page-1'),
        'pages': snapshot.get('whiteboard_pages', ['page-1']),
        'commands': [{key: item.get(key) for key in ('id','kind','payload','page','bounds')}
            for item in snapshot.get('whiteboard', [])]}
    return hashlib.sha256(json.dumps(content, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


class WhiteboardImages:
    def __init__(self):
        self.lock = threading.RLock()
        self.previous = None
        self.revision, self.png, self.digest = None, None, None

    def observe(self, snapshot, bus):
        revision = board_revision(snapshot)
        with self.lock:
            if revision == self.previous: return
            self.previous = revision
        bus.publish('whiteboard.changed', revision=revision)

    def capture(self, revision, png, snapshot, bus):
        from PIL import Image
        if revision != board_revision(snapshot): return False # Reject a late renderer capture.
        if not png.startswith(b'\x89PNG\r\n\x1a\n') or len(png) > 8 * 1024 * 1024: raise ValueError('Invalid board PNG')
        with Image.open(io.BytesIO(png)) as image:
            if image.width > 2048 or image.height > 2048 or image.width < 1 or image.height < 1: raise ValueError('Board capture exceeds size limit')
            image.verify()
        digest = hashlib.sha256(png).hexdigest()
        with self.lock:
            if self.revision == revision and self.digest == digest: return True
            self.revision, self.png, self.digest = revision, png, digest
        bus.publish('whiteboard.image', revision=revision)
        return True

    def image(self, snapshot, root):
        revision = board_revision(snapshot)
        with self.lock:
            if revision == self.revision and self.png: return self.png
        return render_board(snapshot, root)


def render_board(snapshot, root):
    """Bounded simplified fallback when Electron has not supplied a rich capture."""
    from PIL import Image, ImageDraw, ImageFont, ImageColor
    canvas = Image.new('RGB', (1280, 900), '#f7f8fc')
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=20)
    page = snapshot.get('whiteboard_page', 'page-1')
    draw.text((24, 16), f'Whiteboard · {page} (simplified export)', font=font, fill='#172038')
    items = [item for item in snapshot.get('whiteboard', []) if item.get('page', 'page-1') == page]
    if not items:
        draw.text((24, 70), 'The whiteboard is empty.', font=font, fill='#172038')
    shown = items[:256]
    bounds = [item.get('bounds', {}) for item in shown]
    left = min((float(b.get('x', 0)) for b in bounds), default=0)
    top = min((float(b.get('y', 0)) for b in bounds), default=0)
    width = max((float(b.get('x', 0)) + float(b.get('width', 420)) - left for b in bounds), default=420)
    height = max((float(b.get('y', 0)) + float(b.get('height', 120)) - top for b in bounds), default=120)
    scale = min(1, 1220 / max(1, width), 790 / max(1, height))
    for item in shown:
        payload, b = item.get('payload', {}), item.get('bounds', {})
        x, y = 24 + (float(b.get('x', 0)) - left) * scale, 70 + (float(b.get('y', 0)) - top) * scale
        size = max(6, min(128, int(float(payload.get('size', 18)) * scale)))
        try: color = ImageColor.getrgb(payload.get('color') or '#172038')
        except ValueError: color = '#172038'
        if item['kind'] == 'draw':
            points = [(24 + (px-left) * scale, 70 + (py-top) * scale) for px, py in payload.get('points', [])[:10000]]
            if len(points) > 1: draw.line(points, fill=color, width=max(1, size))
            elif points:
                px, py = points[0]; draw.ellipse((px-size/2, py-size/2, px+size/2, py+size/2), fill=color)
        elif item['kind'] == 'image':
            try:
                path = resolve_media(root, payload['path'], extensions={'.png','.jpg','.jpeg','.webp','.gif'})
                if path.stat().st_size > 8 * 1024 * 1024: raise ValueError('Image too large')
                with Image.open(path) as image:
                    if image.width * image.height > 16_000_000: raise ValueError('Image dimensions too large')
                    image = image.convert('RGB')
                    image.thumbnail((max(1, min(1200, int(b.get('width', 420) * scale))), max(1, min(790, int(b.get('height', 120) * scale)))))
                    canvas.paste(image, (int(x), int(y)))
            except (OSError, ValueError, Image.DecompressionBombError):
                draw.text((x, y), '[Image unavailable]', fill=color, font=font)
        else:
            object_font = ImageFont.load_default(size=size)
            columns = max(1, int(float(b.get('width', 420)) * scale / max(1, size * .6)))
            lines = []
            for line in str(payload.get('text', ''))[:12000].splitlines(): lines.extend(textwrap.wrap(line, columns) or [''])
            draw.multiline_text((x, y), '\n'.join(lines[:100]), fill=color, font=object_font, spacing=3)
    if len(items) > len(shown): draw.text((24, 868), f'{len(items)-len(shown)} more objects omitted in simplified export.', fill='#172038', font=font)
    output = io.BytesIO(); canvas.save(output, format='PNG')
    return output.getvalue()
