# Project Riko

A local desktop companion. Electron/React owns presentation; Python independently
owns conversation, microphone capture, transcription, memory, tools and playback.
The character name and personality come from `character_config.yaml`.

## Setup and startup

Install Python runtime dependencies in your virtual environment:

```powershell
.venv\Scripts\python.exe -m pip install -r requirements-runtime.txt
.venv\Scripts\python.exe -m pip install --no-deps EfficientWord-Net
```

`install_reqs.sh` provides the shell-based alternative. Python package metadata
requires 3.11 or newer; model dependencies and CUDA wheels must support your chosen
Python/GPU combination. The current local environment uses Python 3.14. Installing
the requirements alone does not guarantee GPU support. Faster-Whisper GPU use
requires compatible CUDA/cuDNN libraries. Training dependencies are separate in
`requirements-training.txt` and are not needed for desktop inference.

Review `character_config.yaml` before starting:

- `runtime.provider: llama_cpp` launches a separately installed `llama-server`
  executable. Select a local GGUF or an exact Hugging Face file/revision and set
  `runtime.server_path` if the executable is not on PATH. No `llama-cpp-python`
  binding is used. See [model runtime](docs/llama-runtime.md).
- Remote OpenAI-compatible providers, including LM Studio, are optional alternatives.
- Start GPT-SoVITS separately. `sovits_ping_config` configures its HTTP endpoint,
  reference audio/transcript and PCM sample rate. Python does not load its model.
- Voice/ASR, memory, Julia emotion interpretation, avatar and desktop settings are
  configured in YAML or their corresponding frontend settings pages.

Start Python from the repository root:

```powershell
.venv\Scripts\python.exe -u Code\run_server.py
```

In another terminal:

```powershell
cd electron
npm ci
npm run build
npm run start
```

Electron does not start or stop Python. Quit Python with Ctrl+C; the default
Electron quit shortcut is Ctrl+Shift+Q. Default controls/whiteboard shortcuts are
Ctrl+Shift+Space and Ctrl+Shift+W; supported overrides are in desktop settings.

For frontend development, run `npm run dev` in `electron/`, then launch Electron
from a separate terminal with `$env:RIKO_DEV="1"` and `npm run start`.
Ctrl+Shift+I opens control-window DevTools. `desktop.debug: true` enables application
debug logging, not raw microphone/WebSocket packet dumps. Restart to apply YAML changes.

## Current features

- Streamed chat with separate reasoning/tool/runtime activity and interruption history.
- One managed llama-server model, reserved live inference lane and lower-priority
  initiative/reflection slots; native inference uses `/v1/responses`.
- Python-owned microphone, Silero VAD, Faster-Whisper and ordered HTTP TTS playback.
  Microphone capture starts only when requested through Voice controls.
- Durable memories, SQLite conversation archive and revision-controlled tasks.
- Opt-in initiative/event rules with separate observation and spoken permissions.
- Native Electron VRM avatar, transparent effects overlay and persistent paged whiteboard.
- Grouped settings, editable task shortcuts and customizable Appearance.

Settings → Voice output → Response chunking controls the soft word limit,
punctuation priority and search window. The default is 40 words, not a forced
mid-sentence cut. See [speech chunking](docs/speech-chunking.md).

## Voice, avatar and local assets

Voice modes are `wake_word`, `continuous` and `manual`. **Start microphone** enables
capture; **Wake / speak now** activates an utterance. Short wake names use first-run
embedding enrollment through **Set up wake name**. Save repeated samples, test
matches/nonmatches and tune the threshold; profiles are saved per name/microphone
under `persistent_memories/wake_words/`.

Wake activation audio is detector-only: transcription discards pre-activation
buffers and waits for the wake utterance to end. Say the wake name, pause briefly,
then speak the request. Follow-up conversation does not need another keyword.
Manual activation also discards audio captured before the button was pressed.

If GPT-SoVITS is offline, a voice error is shown and the reply continues as text.
Audio is not permanently disabled: each new reply retries synthesis automatically,
so you can start the external server and continue without restarting the app.

Follow-up defaults to 10 seconds after playback. Sustained user speech interrupts
after 1.5 seconds by default; brief interjections preserve context without stopping
capture. Stop overrides speaking priority. Audible text position is estimated,
not word-aligned. Use headphones: speaker echo rejection is not implemented, and
closing an HTTP stream cannot guarantee server-side synthesis cancellation.

Configure `avatar.model` for your compatible VRM. The default path is
`electron/public/models/riko.vrm`; models are local assets, not bundled in Git.
Julia uses normal Hugging Face caching, not a checked-in model directory. VTube
Studio, NDI and face tracking are not used.

Video effects live in `effects/greenscreens/` (`.mp4`, `.webm`, `.mov`, `.m4v`);
playback depends on Electron codec support. Media tools are limited to approved
asset roots. Keep reference audio and other character assets in `character_files/`.

## Python source layout

`Code/process/app_core/` is grouped by responsibility: `configuration/`,
`conversation/`, `inference/`, `runtime/`, `audio/`, `animation/`, `emotion/`,
`persistence/`, `events/`, `resources/`, `tools/` and `desktop/`.
`factory.py` assembles the services; `__init__.py` preserves the public core exports.
Settings, databases, model files and character assets stay in their existing locations.

## Tasks and external MCP

The desktop application registers `TaskMCP` **in-process** in
`Code/process/app_core/factory.py`. Its Tasks panel and model tools share the
configured SQLite store (default `persistent_memories/tasks.sqlite3`). Task contents
are queried when relevant, not automatically injected into every prompt. Updates
require the current revision and retain actor/reason/change history.

`Code/task_mcp_server.py` is an optional **standalone stdio server for other MCP
clients**. The desktop app does not launch it. For an external client:

```json
{
  "mcpServers": {
    "riko-tasks": {
      "command": "C:/path/to/repository/.venv/Scripts/python.exe",
      "args": ["-u", "C:/path/to/repository/Code/task_mcp_server.py", "--store", "C:/path/to/repository/persistent_memories/tasks.sqlite3"]
    }
  }
}
```

Replace the paths; use your configured store if different. Do not add this server
to the companion's own `mcp.json`: its task tools are already registered. External
clients can share the SQLite database. `tests/test_tasks.py` covers the stdio entry
point with a real subprocess round trip. For other tool servers, copy
`mcp.json.example` to `mcp.json` and edit it.

## Optional Discord entry point

`Code/discord_bot.py` is a client of the running Python backend, independent of
the desktop view. You can start this transport explicitly using **Start Discord** beside the chat connection indicator or **Start Discord client** in the taskbar/tray menu. The desktop launch path requires a running backend and local `Discord_bot_token`/`Discord_admins` configuration, does not expose those values to Electron, and reuses its managed process on repeated clicks. The managed client stops with the Python backend; externally started clients remain outside this lifecycle.

The direct command-line client remains independent of
Electron. It does not load a second model. Start `Code/run_server.py` first,
copy the entries from `.env.discord.example` into the repository's `.env`, then run it
with your runtime Python. Keep credentials local and out of version control.
Admin/user/channel lists contain comma-separated numeric Discord IDs. Server
interaction requires a whitelisted text channel **and** a configured trusted user;
DMs require a trusted user. All authorized users share the companion's history,
memory and tools: this is a personal companion, not a privacy-isolated public bot.

Install the optional transport with `pip install -e ".[discord]"`. Enable Message
Content Intent in the Discord Developer Portal and invite with `bot` and
`applications.commands` scopes. Grant View Channel, Send Messages, Read Message
History and Attach Files; Manage Messages is needed only for purging. Calling is
paused; the bot does not require FFmpeg, voice extensions, ASR or GPT-SoVITS.

- `/chat`, `/reasoning`, `/stop`: streamed text replies, separately labelled
  opt-in reasoning and scoped cancellation. Plain messages also start chat; PDF
  attachments provide bounded, labelled excerpts. No audio attachments are sent.
- `/whiteboard`: show the current board image and select this channel for updates.
  Board changes send PNGs to the latest chat/whiteboard channel, driven by events.
  Electron supplies a rich board-only viewport capture when available; otherwise
  the backend produces a simplified active-page export (plain text, images, lines).
- `/join`, `/listen`, `/speak`, `/transcribe`, `/camera` report that calling/audio/
  video are paused pending a custom client. `/leave` safely cleans up leftovers.
- `/settings`, `/tool_policy`, `/status`, `/resources`, `/initiative`, `/animation`:
  narrow admin settings, exact-call approval buttons, runtime/GPU status, optional
  proactive-message relay and desktop animation control. Model/runtime setting
  changes are saved through the existing validator and require a Python restart.
- `/tasks`, `/task_create`, `/task_update`, `/memories`, `/memory`, `/history`,
  `/edit`, `/messages`: durable task/memory controls and confirmed Discord message
  management. Discord message edits/deletions do not erase companion memory/history.
- No camera or screen feed is captured or sent through Discord. Preserved calling
  code is disabled and no self-bot/user-token workaround is used.

Discord transport preferences are saved under
`persistent_memories/discord_preferences.json`; voice consent is not persisted.
Only the bot token is required to launch the transport, not in tests/imports;
configured admin IDs are also required for safe operation. See
[`docs/discord-integration.md`](docs/discord-integration.md) for limits and verification.

## Documentation and checks

- [Current status and limitations](docs/progress.md)
- [Independent settings, foreground priority and desktop speech feedback](docs/desktop-feedback.md)
- [Glass appearance, window controls and microphone dock](docs/frontend-experience.md)
- [Floating chat, mic status and whiteboard controls](docs/floating-chat.md)
- [Source ownership and repository review](docs/project-review.md)
- [Model settings, slots and caching](docs/llama-runtime.md)
- [Desktop settings and appearance](docs/desktop-settings.md)
- [Speech chunking](docs/speech-chunking.md)
- [Wake acknowledgement sounds and animations](docs/wake-feedback.md)
- [Animation engine, VRMA import and desktop interactions](docs/animation-engine.md)
- [Animation design and neural-policy roadmap](docs/animation-engine-plan.md)
- [History, persistence, deadlines and live checks](docs/runtime-recovery.md)
- [Product requirements](docs/requirements.md) and [remaining work](docs/implementation-plan.md)
- [Memory design notes](docs/memory-design-notes.md) — design background, not current implementation status

```powershell
.venv\Scripts\python.exe -m pytest -q
```

From `electron/`: `npm test`, `npm run build`, `node --check main.cjs` and
`node --check preload.cjs`. Automated tests do not establish real microphone,
GPU throughput or desktop-device behavior; consult the live-check guides.

Generated dependencies/builds/models are ignored. `persistent_memories/` contains
user data and must not be deleted during cleanup. Existing staged artifact removals
do not erase blobs from historical commits; see the repository review.

## Credits

- Inspired by [Ryan's Riko Project](https://github.com/rayenfeng/riko_project).
- Voice synthesis: [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS).
- ASR: [Faster-Whisper](https://github.com/SYSTRAN/faster-whisper).
