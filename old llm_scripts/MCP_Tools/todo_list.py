import json
import pathlib
from typing import Any, Dict, List, Optional
from datetime import datetime

from .base import BaseTool, ToolType


class Tool(BaseTool):
    TOOL_NAME = "todo_list"
    TOOL_DESCRIPTION = "Manage a persistent to-do list: add, remove, list, complete, and clear tasks."
    TOOL_TYPE = ToolType.FUNCTION

    MCP_PROMPT = """todo_list:
  Manage a persistent to-do list. All tasks are saved automatically.

  Actions:
    list                          - Show all tasks with their status.
    add <task description>        - Add a new task (e.g., "add Buy groceries").
    remove <task_id>              - Remove a task by its ID (shown in list).
    complete <task_id>            - Mark a task as done (toggle).
    clear                         - Delete all tasks.

  Task IDs are shown in the list output (e.g., "[1] Buy milk").
  Tasks are stored in './persistent_memories/mcp_modules/todo_list/tasks.json'.

  Examples:
    todo_list(action="list")                      -> Shows all tasks.
    todo_list(action="add", task="Write report")  -> Adds a new task.
    todo_list(action="complete", task_id="2")     -> Toggles task #2 as done.
    todo_list(action="remove", task_id="3")       -> Deletes task #3.
    todo_list(action="clear")                     -> Removes all tasks.
"""

    # Persistent storage
    DATA_DIR = pathlib.Path("./persistent_memories/mcp_modules/todo_list")
    DATA_FILE = DATA_DIR / "tasks.json"

    def _ensure_storage(self) -> None:
        """Create the storage directory if it doesn't exist."""
        self.DATA_DIR.mkdir(parents=True, exist_ok=True)

    def _load_tasks(self) -> List[Dict[str, Any]]:
        """Load tasks from the JSON file, or return an empty list."""
        self._ensure_storage()
        if not self.DATA_FILE.exists():
            return []
        try:
            with open(self.DATA_FILE, "r", encoding="utf-8") as f:
                tasks = json.load(f)
                if isinstance(tasks, list):
                    return tasks
                else:
                    return []
        except (json.JSONDecodeError, IOError):
            return []

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
            json.dump(tasks, tmp, indent=2)
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
            for idx, t in enumerate(tasks, start=1):
                status = "✅" if t.get("done", False) else "⬜"
                lines.append(f"{idx}. {status} {t['text']}")
            return "\n".join(lines)

        elif action == "add":
            if not task or not task.strip():
                return "❌ Please provide a task description (e.g., `add Buy milk`)."
            tasks.append({
                "text": task.strip(),
                "done": False,
                "created_at": datetime.now().isoformat(timespec="minutes")
            })
            self._save_tasks(tasks)
            return f"✅ Added task: '{task.strip()}' (ID: {len(tasks)})"

        elif action == "complete":
            if task_id is None or not task_id.strip():
                return "❌ Please provide a task ID (e.g., `complete 2`)."
            try:
                idx = int(task_id.strip()) - 1
                if idx < 0 or idx >= len(tasks):
                    return f"❌ Task ID {task_id} not found."
                tasks[idx]["done"] = not tasks[idx].get("done", False)
                status = "done" if tasks[idx]["done"] else "undone"
                self._save_tasks(tasks)
                return f"✅ Task {task_id} marked as {status}."
            except ValueError:
                return f"❌ Invalid task ID: '{task_id}'. Please use a number."

        elif action == "remove":
            if task_id is None or not task_id.strip():
                return "❌ Please provide a task ID (e.g., `remove 2`)."
            try:
                idx = int(task_id.strip()) - 1
                if idx < 0 or idx >= len(tasks):
                    return f"❌ Task ID {task_id} not found."
                removed = tasks.pop(idx)
                self._save_tasks(tasks)
                return f"✅ Removed task: '{removed['text']}'"
            except ValueError:
                return f"❌ Invalid task ID: '{task_id}'. Please use a number."

        elif action == "clear":
            if not tasks:
                return "📭 Your to‑do list is already empty."
            self._save_tasks([])
            return "🗑️ All tasks have been cleared."

        else:
            return f"❌ Unknown action: '{action}'. Available: list, add, complete, remove, clear."

    # ------------------------------------------------------------------
    # (Optional) Initialisation: ensure directory exists
    # ------------------------------------------------------------------
    def _setup(self):
        """Ensure the storage directory exists on startup."""
        self._ensure_storage()