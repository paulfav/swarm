from __future__ import annotations

import json
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
        self._llm_paths: dict[str, str] = {}

    def register_agent(self, drone_id: str) -> str:
        path = os.path.join(self.run_dir, f"{drone_id}.txt")
        # ensure file exists
        with open(path, "w", encoding="utf-8") as f:
            f.write("")
        self._agent_paths[drone_id] = path
        return path

    def register_llm_log(self, drone_id: str) -> str:
        path = os.path.join(self.run_dir, f"{drone_id}_llm.txt")
        # ensure file exists
        with open(path, "w", encoding="utf-8") as f:
            f.write("")
        self._llm_paths[drone_id] = path
        return path

    def log_llm_conversation(self, drone_id: str, step: int, system_prompt: str, user_input: str, response: dict, tool_calls: List[dict] = None) -> None:
        path = self._llm_paths.get(drone_id)
        if path is None:
            path = self.register_llm_log(drone_id)
        
        log_entry = {
            "step": step,
            "timestamp": datetime.now().isoformat(),
            "system_prompt": system_prompt,
            "user_input": user_input,
            "response": response,
            "tool_calls": tool_calls or []
        }
        
        with open(path, "a", encoding="utf-8") as f:
            f.write(f"=== STEP {step} ===\n")
            f.write(f"System: {system_prompt}\n")
            f.write(f"User: {user_input}\n")
            f.write(f"Response: {json.dumps(response, indent=2)}\n")
            if tool_calls:
                f.write(f"Tool Calls: {json.dumps(tool_calls, indent=2)}\n")
            f.write("\n")

    def log_strategy_report(self, report: str) -> None:
        """Log strategy reports to a separate file"""
        path = os.path.join(self.run_dir, "strategy_reports.txt")
        with open(path, "a", encoding="utf-8") as f:
            f.write(f"{report}\n")
            f.write("=" * 80 + "\n\n")

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