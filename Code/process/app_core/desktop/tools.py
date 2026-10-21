from __future__ import annotations

from .effects import EffectsModel
from .geometry import SurfaceGeometry
from .media import EFFECTS_DIRECTORY
from .state import get_desktop_state
from .whiteboard import PAGE_ACTIONS, WhiteboardModel
from dataclasses import dataclass
import math
import re
from pathlib import Path
from typing import Any, Callable


@dataclass(slots=True)
class DesktopServices:
    """What the desktop tools act through, handed to them by the composition root (factory.create_chat_service): the desktop
    components they change and the collaborators they use. DesktopServices.of(state) takes a DesktopState's components; a
    tool built without services uses get_desktop_state()'s and has no media, effect library, gestures or walking."""
    board: WhiteboardModel
    geometry: SurfaceGeometry
    effects: EffectsModel
    actions: Any = None  # the ActionController that plays avatar gestures
    media_resolver: Callable[[str], Path] | None = None  # resolve_media bound to the data root: approved roots and extensions
    effects_directory: str = EFFECTS_DIRECTORY
    effect_library: Any = None  # EffectLibrary over effects_directory
    avatar_motion: Any = None  # the session's AnimationRuntime once it runs (SessionManager attaches it): move_avatar walks

    @classmethod
    def of(cls, state, **collaborators): return cls(state.board, state.geometry, state.effects, **collaborators)


class DesktopTool:
    """A tool run in-process (tools.tool.local_tool): execute's signature is its input schema."""
    def __init__(self, services: DesktopServices | None = None):
        self.services = services or DesktopServices.of(get_desktop_state())


class WhiteboardTool(DesktopTool):
    TOOL_NAME = "whiteboard"
    TOOL_DESCRIPTION = "Place Markdown/LaTeX text, drawing or image on a paged whiteboard. Omit x/y for nonoverlapping automatic placement. Returns measured logical rectangle bounds when renderer responds, otherwise estimated queued bounds. Actions: text, draw, image, clear, pages, new_page, page, next_page, previous_page. User pan/zoom/moves are local and do not change model layout."
    CHOICES = {'action':['text','draw','image','clear','pages','new_page','page','next_page','previous_page']}

    def execute(self, action: str, text: str = "", points: list | None = None, image_path: str = "", x: int | None = None, y: int | None = None, color: str = "#ffffff", size: int = 18, width: int = 420, page: str | None = None) -> str:
        if action in PAGE_ACTIONS: return str(self.services.board.pages(action, page))
        if action == "clear": return f'Whiteboard clear queued: {self.services.board.clear()}. Await renderer acknowledgement.'
        def number(value): return type(value) in (int, float) and math.isfinite(value)
        if not all(value is None or number(value) for value in (x, y)) or not number(size) or not 1 <= size <= 128 or not number(width) or not 100 <= width <= 2000: raise ValueError('Invalid position/size')
        if not isinstance(color, str) or not re.fullmatch(r'#[0-9a-fA-F]{6}', color): raise ValueError('Use a six-digit hex color')
        if action == 'text':
            if not isinstance(text, str) or not text.strip(): raise ValueError('Text is required')
            payload = {'text': text, 'x': x, 'y': y, 'color': color, 'size': size, 'width': width}
        elif action == 'draw':
            if not isinstance(points, list) or not 1 <= len(points) <= 10000: raise ValueError('Provide 1–10000 points')
            normalized = []
            for point in points:
                values = [point.get('x'), point.get('y')] if isinstance(point, dict) else point
                if not isinstance(values, (list, tuple)) or len(values) != 2 or not all(number(v) for v in values): raise ValueError('Points must be [x,y] pairs or {x,y} objects')
                normalized.append(values)
            payload = {'points': normalized, 'color': color, 'size': size}
        elif action == 'image':
            resolver = self.services.media_resolver
            if not resolver: raise RuntimeError('Media resolver unavailable')
            asset = resolver(image_path)
            if asset.suffix.lower() not in {'.png', '.jpg', '.jpeg', '.gif', '.webp'}: raise ValueError('Whiteboard requires an image')
            payload = {'path': str(asset), 'x': x, 'y': y, 'width': width}
        else: raise ValueError('Use text, draw, image or clear')
        command_id = self.services.board.add(action, payload)
        return str(self.services.board.result(command_id))


class AvatarWindowTool(DesktopTool):
    TOOL_NAME = "move_avatar"
    TOOL_DESCRIPTION = "Move the VRM avatar with its walk cycle by default; resize/reposition its display using width/height/screen. Set walk=false only for an explicit teleport. Coordinates are logical display pixels. User dragging overrides walking. Effects follow the selected display."

    def execute(self, x: int | None = None, y: int | None = None, width: int | None = None, height: int | None = None, screen: int | None = None, walk: bool = True) -> str:
        if type(walk) is not bool: raise ValueError('walk must be boolean')
        if any(value is not None and type(value) is not int for value in (x,y,width,height,screen)): raise ValueError('Avatar placement must use integer pixels/display indices')
        runtime = self.services.avatar_motion
        geometry = {key:value for key,value in {'width':width,'height':height,'screen':screen}.items() if value is not None}
        if geometry:
            if runtime: runtime.stop_movement()
            self.services.geometry.update('avatar', **geometry)
        if x is not None or y is not None:
            if walk and runtime:
                current = self.services.geometry.avatar()
                action = runtime.walk_to(current['x'] if x is None else x, current['y'] if y is None else y)
                return f'Walk queued: {action.id}. Renderer reports completion; use runtime_status for placement.'
            self.services.geometry.update('avatar', **{key:value for key,value in {'x':x,'y':y}.items() if value is not None})
        return 'Avatar geometry updated.'


class WhiteboardWindowTool(DesktopTool):
    TOOL_NAME = "move_whiteboard"
    TOOL_DESCRIPTION = "Move or resize the companion whiteboard across desktop displays."

    def execute(self, x: int | None = None, y: int | None = None, width: int | None = None, height: int | None = None, screen: int | None = None) -> str:
        values = {k: v for k, v in locals().items() if k != "self" and v is not None}
        self.services.geometry.update("whiteboard", **values); return "Whiteboard geometry updated."


class EffectTool(DesktopTool):
    TOOL_NAME = "visual_effect"
    TOOL_DESCRIPTION = "List available local video effects, play one by filename/rule name, or stop it. Playback success is confirmed by renderer acknowledgement, not the queued result."
    CHOICES = {'action':['list','play','stop']}

    def input_choices(self, arguments=None):
        """The actions and effect names, but only the action for a call that lists, stops or names an asset; an empty name
        is the default the model filled in, not a choice to repair."""
        if arguments and (arguments.get('asset') or arguments.get('action') in {'list','stop'}): return dict(self.CHOICES)
        library = self.services.effect_library
        if not library or (arguments and arguments.get('name') == ''): return dict(self.CHOICES)
        names = {name for paths in library.assets.values() for path in paths for name in (path.name,path.stem)}
        names.update(rule.name for rule in library.rules if rule.enabled and rule.asset)
        return {**self.CHOICES, 'name': sorted(names)}
    def prepare_arguments(self, arguments):
        asset = arguments.get('asset')
        if arguments.get('action','play') != 'play': return arguments, []
        resolver, library = self.services.media_resolver, self.services.effect_library
        if not resolver: raise ValueError('Media resolver unavailable')
        original = asset
        if not asset:
            name = arguments.get('name','')
            if not isinstance(name,str) or Path(name).name != name or not library: raise ValueError('Choose a listed effect name or approved asset')
            rules = [rule for rule in library.rules if rule.enabled and rule.name == name and rule.asset]
            matches = [Path(rule.asset) if Path(rule.asset).is_absolute() else library.directory / rule.asset for rule in rules]
            if not matches: matches = [path for paths in library.assets.values() for path in paths if name in {path.name,path.stem}]
            matches = list(dict.fromkeys(matches))
            if len(matches) != 1: raise ValueError('Effect name is missing or ambiguous. Use visual_effect action=list and an exact asset path')
            asset = str(matches[0])
        try: resolved = resolver(asset)
        except ValueError as exc:
            if not isinstance(asset,str) or Path(asset).is_absolute() or '..' in asset.replace('\\','/').split('/') or ':' in asset or not library: raise exc
            matches = [path for paths in library.assets.values() for path in paths
                if asset.replace('\\','/').casefold() in {path.name.casefold(),path.relative_to(library.directory).as_posix().casefold()}]
            if len(matches) != 1: raise ValueError('Choose an exact available effect asset using visual_effect action=list') from exc
            resolved = resolver(str(matches[0])) # Revalidate permission boundary; no directory widening.
        if resolved.suffix.lower() not in {'.mp4','.webm','.mov','.m4v'}: raise ValueError('Effects require a video')
        effective = {**arguments,'asset':str(resolved)}
        corrections = [] if str(resolved) == original else [{'parameter':'asset','from':original,'to':str(resolved)}]
        return effective, corrections
    def execute(self, action: str = "play", name: str = "", asset: str | None = None, opacity: float = 0.65, brightness: float = 1.0, duration: float = 8.0) -> str:
        library = self.services.effect_library
        if action == 'list':
            if not library: return 'Effect library unavailable'
            return str({'assets': [str(path.relative_to(library.directory)) for paths in library.assets.values() for path in paths if path.suffix.lower() in {'.mp4', '.webm', '.mov', '.m4v'}],
                        'rules': library.rules_json()})
        if action == "stop": self.services.effects.stop(); return "Visual effect stopped."
        if action != 'play': raise ValueError('Use play or stop')
        if not all(isinstance(v, (float, int)) and math.isfinite(v) for v in (opacity, brightness, duration)) or not 0 <= opacity <= 1 or not 0 <= brightness <= 4 or not .1 <= duration <= 300: raise ValueError('Invalid effect opacity, brightness or duration')
        resolver = self.services.media_resolver
        if not resolver: raise RuntimeError('Media resolver unavailable')
        if not asset:
            if not name or Path(name).name != name: raise ValueError('Provide an effect filename from visual_effect action=list, or an approved asset path')
            rule = next((rule for rule in library.rules if rule.enabled and rule.name == name and rule.asset), None) if library else None
            if rule: asset = str(Path(rule.asset) if Path(rule.asset).is_absolute() else library.directory / rule.asset)
            else:
                found = library.find_asset(name) if library else None
                asset = str(found) if found else str(Path(self.services.effects_directory) / name)
        resolved = resolver(asset)
        if resolved.suffix.lower() not in {'.mp4', '.webm', '.mov', '.m4v'}: raise ValueError('Effects require a video')
        command = self.services.effects.trigger(name or resolved.name, opacity=opacity, brightness=brightness, duration=duration, asset=str(resolved))
        return f'Effect queued: {command}. Await playback acknowledgement.'


class AvatarGestureTool(DesktopTool):
    TOOL_NAME = "avatar_gesture"
    TOOL_DESCRIPTION = "Perform a nod, shake or wave gesture, or cancel an action by its returned ID."
    CHOICES = {'name':['nod','shake','wave']}

    def input_choices(self, arguments=None):
        # A cancel needs no gesture, and an empty name is the default the model filled in, not a choice to repair.
        return {} if arguments and (arguments.get('cancel_id') or arguments.get('name') == '') else dict(self.CHOICES)

    def execute(self, name: str = "nod", intensity: float = 0.65, duration: float = 2.0, cancel_id: str = "") -> str:
        controller = self.services.actions
        if controller is None:
            raise RuntimeError("Avatar action controller unavailable")
        if cancel_id:
            return "Cancelled" if controller.cancel(cancel_id) else "Action no longer active"
        action = controller.gesture(name, intensity, duration)
        return f"Gesture scheduled: {action.id}"


def iter_tools(services: DesktopServices | None = None):
    return [tool(services) for tool in (WhiteboardTool, AvatarWindowTool, WhiteboardWindowTool, EffectTool, AvatarGestureTool)]
