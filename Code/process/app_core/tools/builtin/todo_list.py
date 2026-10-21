import json
import pathlib
from typing import Any, Dict, List, Optional
from datetime import datetime

from .base import BaseTool


class Tool(BaseTool):
    TOOL_NAME = "todo_list"
    TOOL_DESCRIPTION = "Manage a persistent to-do list: add, remove, list, complete, and clear tasks."

    # Persistent storage: config['data_directory'] (DataPaths.todo_list, from the registry); standalone, the cwd's folder.
    DATA_DIR = pathlib.Path("./persistent_memories/mcp_modules/todo_list")
    DATA_FILE = DATA_DIR / "tasks.json"

    def _ensure_storage(self) -> None:
        """Create the storage directory if it doesn't exist."""
        self.DATA_DIR.mkdir(parents=True, exist_ok=True)

    def _load_tasks(self) -> List[Dict[str, Any]]:
        """Load tasks. An unreadable list is kept aside and reported, never silently replaced."""
        self._ensure_storage()
        self._next_id = 1
        if not self.DATA_FILE.exists():
            return []
        try:
            with open(self.DATA_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list): data = {"tasks": data}  # Saved before the ID counter existed.
            tasks = data.get("tasks") if isinstance(data, dict) else None
            if not isinstance(tasks, list) or not all(isinstance(t, dict) and isinstance(t.get("text"), str) for t in tasks):
                raise ValueError("not a task list")
        except OSError as exc:
            raise RuntimeError(f"The to-do list could not be read ({exc}); nothing was changed.") from exc
        except ValueError as exc:
            backup = self.DATA_FILE.with_name(f"tasks.json.unreadable-{datetime.now():%Y%m%d-%H%M%S-%f}")
            self.DATA_FILE.replace(backup)
            raise RuntimeError(f"The to-do list file was unreadable and was kept as {backup.name}; "
                               "a new list starts with the next change.") from exc
        # Stable IDs that are never reused: the counter survives removals and clear. Lists saved
        # before IDs existed are numbered in their current order; duplicate IDs get fresh ones.
        next_id = max((t["id"] for t in tasks if type(t.get("id")) is int), default=0) + 1
        if type(data.get("next_id")) is int: next_id = max(next_id, data["next_id"])
        seen = set()
        for t in tasks:
            if type(t.get("id")) is not int or t["id"] in seen:
                t["id"], next_id = next_id, next_id + 1
            seen.add(t["id"])
        self._next_id = next_id
        return tasks

    @staticmethod
    def _find(tasks: List[Dict[str, Any]], task_id: Optional[str]) -> Dict[str, Any]:
        if task_id is None or not str(task_id).strip():
            raise ValueError("Please provide a task ID from the list (e.g., task_id=\"2\").")
        try: wanted = int(str(task_id).strip())
        except ValueError: raise ValueError(f"Invalid task ID: '{task_id}'. Please use the number shown in the list.") from None
        for t in tasks:
            if t["id"] == wanted: return t
        raise ValueError(f"Task ID {task_id} not found.")

    def _save_tasks(self, tasks: List[Dict[str, Any]]) -> None:
        """Save tasks to the JSON file atomically (via temp file)."""
        import tempfile
        import os

        self._ensure_storage()
        with tempfile.NamedTemporaryFile(
            "w",
            dir=self.DATA_DIR,
            delete=False,
            encoding="utf-8"
        ) as tmp:
            json.dump({"next_id": self._next_id, "tasks": tasks}, tmp, indent=2)
            tmp.flush()
            os.fsync(tmp.fileno())
            temp_path = pathlib.Path(tmp.name)
        temp_path.replace(self.DATA_FILE)

    # Core tool logic
    def _call(self, action: str, task: Optional[str] = None, task_id: Optional[str] = None) -> str: #type: ignore
        """
        Execute the requested action on the to‑do list.
        Returns a human‑readable result string.
        """
        tasks = self._load_tasks()

        # Normalise action
        action = action.lower().strip()

        if action == "list":
            if not tasks:
                return "📭 Your to‑do list is empty."
            lines = ["📋 **Your To‑Do List:**"]
            for t in tasks:
                status = "✅" if t.get("done", False) else "⬜"
                lines.append(f"{t['id']}. {status} {t['text']}")
            return "\n".join(lines)

        elif action == "add":
            if not task or not task.strip():
                raise ValueError("Please provide a task description (e.g., task=\"Buy milk\").")
            new_id, self._next_id = self._next_id, self._next_id + 1
            tasks.append({
                "id": new_id,
                "text": task.strip(),
                "done": False,
                "created_at": datetime.now().isoformat(timespec="minutes")
            })
            self._save_tasks(tasks)
            return f"✅ Added task: '{task.strip()}' (ID: {new_id})"

        elif action == "complete":
            found = self._find(tasks, task_id)
            found["done"] = not found.get("done", False)
            self._save_tasks(tasks)
            return f"✅ Task {found['id']} marked as {'done' if found['done'] else 'undone'}."

        elif action == "remove":
            found = self._find(tasks, task_id)
            tasks.remove(found)
            self._save_tasks(tasks)
            return f"✅ Removed task {found['id']}: '{found['text']}'"

        elif action == "clear":
            if not tasks:
                return "📭 Your to‑do list is already empty."
            self._save_tasks([])
            return "🗑️ All tasks have been cleared."

        else:
            raise ValueError(f"Unknown action: '{action}'. Available: list, add, complete, remove, clear.")

    # ------------------------------------------------------------------
    # (Optional) Initialisation: ensure directory exists
    # ------------------------------------------------------------------
    def _setup(self):
        """Use the configured folder, and ensure the storage directory exists on startup."""
        if self.config.get("data_directory"):
            self.DATA_DIR = pathlib.Path(self.config["data_directory"])
            self.DATA_FILE = self.DATA_DIR / "tasks.json"
        self._ensure_storage()
