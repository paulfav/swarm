from __future__ import annotations

import os
from datetime import datetime
from typing import List, Optional


class RunLogger:
    def __init__(self, base_dir: Optional[str] = None, run_dir: Optional[str] = None) -> None:
        base = base_dir or os.path.join(os.getcwd(), "log")
        os.makedirs(base, exist_ok=True)
        if run_dir is None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_dir = f"run_{ts}"
        self.run_dir = os.path.join(base, run_dir)
        os.makedirs(self.run_dir, exist_ok=True)
        self._agent_paths: dict[str, str] = {}

    def register_agent(self, drone_id: str) -> str:
        path = os.path.join(self.run_dir, f"{drone_id}.txt")
        # ensure file exists
        with open(path, "w", encoding="utf-8") as f:
            f.write("")
        self._agent_paths[drone_id] = path
        return path

    def flush_agent_memory(self, drone_id: str, memory: List[str], max_lines: int = 2000) -> None:
        path = self._agent_paths.get(drone_id)
        if path is None:
            path = self.register_agent(drone_id)
        # keep only the last max_lines lines to avoid huge files
        lines = memory[-max_lines:]
        with open(path, "w", encoding="utf-8") as f:
            for line in lines:
                # hard cap line length
                safe_line = line[:2000]
                f.write(f"{safe_line}\n") 