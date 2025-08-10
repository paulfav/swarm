from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional
import numpy as np

from core.environment import Environment
from core.swarm import Swarm
from tools import movement

try:
    from openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore


ToolFunc = Callable[..., Dict]


def get_llm_provider() -> str:
    return os.getenv("LLM_PROVIDER", "mock").lower()


@dataclass
class DroneAgent:
    drone_id: str
    env: Environment
    swarm: Swarm
    tools: Dict[str, ToolFunc] = field(default_factory=dict)
    memory: List[str] = field(default_factory=list)
    memory_limit: int = 100
    home_position: tuple[int, int, int] | None = None
    sector_bounds: tuple[int, int, int, int] | None = None  # x0, x1, y0, y1
    phase: str = "explore"  # explore | return
    step_count: int = 0

    def __post_init__(self) -> None:
        if not self.tools:
            self.tools = {
                "move": movement.move,
                "scan": movement.scan,
                "move_delta": movement.move_delta,
                "send_message": self._tool_send_message,
            }

        provider = get_llm_provider()
        if provider == "openai" and OpenAI is not None:
            self._client = OpenAI()
        else:
            self._client = None

        # establish home and sector
        self.home_position = self.env.get_drone_position(self.drone_id)
        self.sector_bounds = self._compute_sector_bounds()
        # announce sector
        try:
            x0, x1, y0, y1 = self.sector_bounds
            self._tool_send_message(
                recipient_id=None,
                message=f"Claiming sector x[{x0},{x1}], y[{y0},{y1}]"
            )
        except Exception:
            pass

    def _compute_sector_bounds(self) -> tuple[int, int, int, int]:
        # Divide X range into equal vertical slices by drone index
        sx, sy = self.env.size_x, self.env.size_y
        ids = self.swarm.drone_ids
        n = max(1, len(ids))
        idx = max(0, ids.index(self.drone_id)) if self.drone_id in ids else 0
        slice_width = sx // n
        x0 = idx * slice_width
        x1 = (idx + 1) * slice_width - 1 if idx < n - 1 else sx - 1
        y0, y1 = 0, sy - 1
        return (x0, x1, y0, y1)

    def _nearest_unscanned_in_sector(self) -> Optional[tuple[int, int, int]]:
        x0, x1, y0, y1 = self.sector_bounds if self.sector_bounds else (0, self.env.size_x - 1, 0, self.env.size_y - 1)
        scanned = self.env.scanned
        xs, ys, zs = np.where(scanned == 0)
        # mask to sector bounds
        mask = (xs >= x0) & (xs <= x1) & (ys >= y0) & (ys <= y1)
        xs, ys, zs = xs[mask], ys[mask], zs[mask]
        if xs.size == 0:
            return None
        cx, cy, cz = self.env.get_drone_position(self.drone_id)
        # prioritize same Z, then closest by L1 distance
        dz = np.abs(zs - cz)
        # weight z difference slightly higher to favor same altitude
        distances = np.abs(xs - cx) + np.abs(ys - cy) + (dz * 2)
        i = int(np.argmin(distances))
        return (int(xs[i]), int(ys[i]), int(zs[i]))

    def _coverage_in_sector(self) -> float:
        x0, x1, y0, y1 = self.sector_bounds if self.sector_bounds else (0, self.env.size_x - 1, 0, self.env.size_y - 1)
        scanned_slice = self.env.scanned[x0:x1 + 1, y0:y1 + 1, :]
        return float(scanned_slice.sum()) / float(scanned_slice.size)

    # Tool wrappers
    def _tool_send_message(self, recipient_id: Optional[str] = None, message: str = "") -> Dict:
        self.swarm.send_message(sender_id=self.drone_id, recipient_id=recipient_id, content=message)
        return {"ok": True}

    # Prompt assembly
    def _build_system_prompt(self) -> str:
        return (
            "You are a search-and-rescue drone. You can move, scan, and communicate with peers. "
            "Your goal is to find all missing targets efficiently. "
            "Tools available: move(direction), scan(radius), send_message(drone_id|None, message). "
            "Decide one action per step. Prefer scanning when in new areas; coordinate to divide the space. "
            "Use local_unscanned_counts and suggested_direction to choose where to move to explore unscanned space."
        )

    def _compute_unscanned_counts(self, radius: int = 3) -> Dict[str, int]:
        cx, cy, cz = self.env.get_drone_position(self.drone_id)
        sx, sy, sz = self.env.size_x, self.env.size_y, self.env.size_z
        counts = {k: 0 for k in ["north", "south", "east", "west", "up", "down"]}
        x0 = max(0, cx - radius)
        x1 = min(sx - 1, cx + radius)
        y0 = max(0, cy - radius)
        y1 = min(sy - 1, cy + radius)
        z0 = max(0, cz - radius)
        z1 = min(sz - 1, cz + radius)
        scanned = self.env.scanned
        for x in range(x0, x1 + 1):
            for y in range(y0, y1 + 1):
                for z in range(z0, z1 + 1):
                    if scanned[x, y, z] == 0:  # unscanned
                        if y > cy:
                            counts["north"] += 1
                        elif y < cy:
                            counts["south"] += 1
                        if x > cx:
                            counts["east"] += 1
                        elif x < cx:
                            counts["west"] += 1
                        if z > cz:
                            counts["up"] += 1
                        elif z < cz:
                            counts["down"] += 1
        return counts

    def _suggest_direction_from_counts(self, counts: Dict[str, int]) -> str:
        # pick the direction with the maximum count, fallback to north
        if not counts:
            return "north"
        return max(counts.items(), key=lambda kv: kv[1])[0]

    def _build_state_snapshot(self) -> str:
        pos = self.env.get_drone_position(self.drone_id)
        remaining = [t.id for t in self.env.remaining_targets()]
        discovered_targets = self.env.discovered_targets()
        discovered = [t.id for t in discovered_targets]
        inbox = self.swarm.get_messages(self.drone_id)
        msgs = [f"from {m.sender_id}: {m.content}" for m in inbox]
        coverage = float(self.env.scanned.sum()) / float(self.env.scanned.size)
        recent_actions = self.memory[-5:]
        recent_actions = [a[:200] for a in recent_actions]
        local_counts = self._compute_unscanned_counts(radius=3)
        suggested_direction = self._suggest_direction_from_counts(local_counts)

        # nearest discovered target (if any)
        nearest_target_pos = None
        if discovered_targets:
            cx, cy, cz = pos
            best_dist = None
            for t in discovered_targets:
                tx, ty, tz = t.position
                dist = abs(tx - cx) + abs(ty - cy) + abs(tz - cz)
                if best_dist is None or dist < best_dist:
                    best_dist = dist
                    nearest_target_pos = t.position

        nearest_unscanned = self._nearest_unscanned_in_sector()
        sector_cov = self._coverage_in_sector()

        return json.dumps({
            "drone_id": self.drone_id,
            "position": pos,
            "coverage": coverage,
            "remaining_targets": remaining,
            "discovered_targets": discovered,
            "messages": msgs,
            "recent_actions": recent_actions,
            "local_unscanned_counts": local_counts,
            "suggested_direction": suggested_direction,
            "nearest_discovered_target": nearest_target_pos,
            "home_position": self.home_position,
            "phase": self.phase,
            "sector_bounds": self.sector_bounds,
            "sector_coverage": sector_cov,
            "nearest_unscanned_in_sector": nearest_unscanned,
        })

    def _direction_toward(self, target_pos: tuple[int, int, int], current: tuple[int, int, int]) -> str:
        cx, cy, cz = current
        tx, ty, tz = target_pos
        if tx > cx:
            return "east"
        if tx < cx:
            return "west"
        if ty > cy:
            return "north"
        if ty < cy:
            return "south"
        if tz > cz:
            return "up"
        if tz < cz:
            return "down"
        return "north"

    # Decision step
    def step(self) -> Dict[str, Any]:
        provider = get_llm_provider()
        self.step_count += 1
        state = self._build_state_snapshot()
        system = self._build_system_prompt()

        if provider == "openai" and self._client is not None:
            return self._openai_decide_and_act(system, state)
        else:
            return self._mock_decide_and_act(system, state)

    # OpenAI action selection via tool-calling
    def _openai_decide_and_act(self, system_prompt: str, state_json: str) -> Dict[str, Any]:
        tool_defs = [
            {
                "type": "function",
                "function": {
                    "name": "move",
                    "description": "Move the drone in a cardinal or diagonal direction",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "direction": {"type": "string"}
                        },
                        "required": ["direction"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "scan",
                    "description": "Scan the area around the drone with a given radius",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "radius": {"type": "integer", "minimum": 1, "maximum": 6}
                        },
                        "required": ["radius"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "send_message",
                    "description": "Send a message to a specific drone or broadcast if recipient is null",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "recipient_id": {"type": ["string", "null"]},
                            "message": {"type": "string"}
                        },
                        "required": ["message"],
                    },
                },
            },
        ]

        resp = self._client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            temperature=0.2,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"STATE: {state_json}"},
                {"role": "user", "content": "Choose ONE action by calling a tool."},
            ],
            tools=tool_defs,
            tool_choice="auto",
        )
        choice = resp.choices[0]
        tool_calls = getattr(choice.message, "tool_calls", None) or []
        if not tool_calls:
            name = "scan"
            args = {"radius": 2}
        else:
            call = tool_calls[0]
            name = call.function.name
            args = json.loads(call.function.arguments or "{}")
        result = self._execute_action(name, args)
        # compact log line without embedding prior memory/state
        pos = self.env.get_drone_position(self.drone_id)
        args_short = str(args)[:120]
        res_short = json.dumps(result)[:200]
        self.memory.append(f"pos={pos} action={name} args={args_short} result={res_short}")
        if len(self.memory) > self.memory_limit:
            self.memory = self.memory[-self.memory_limit:]
        return result

    # Mock policy: simple random walk + periodic scan + occasional broadcast
    def _mock_decide_and_act(self, system_prompt: str, state_json: str) -> Dict[str, Any]:
        rnd = random.Random(self.drone_id + str(len(self.memory)))
        pos = self.env.get_drone_position(self.drone_id)

        # Phase switching: return when high overall coverage or all targets found
        all_found = len(self.env.remaining_targets()) == 0
        overall_coverage = float(self.env.scanned.sum()) / float(self.env.scanned.size)
        sector_cov = self._coverage_in_sector()
        if self.phase == "explore" and (all_found or sector_cov >= 0.9 or overall_coverage >= 0.9):
            self.phase = "return"

        if self.phase == "return":
            # move toward home base
            direction = self._direction_toward(self.home_position, pos) if self.home_position else "north"
            action_name = "move"
            action_args = {"direction": direction}
            result = self._execute_action(action_name, action_args)
            # log
            args_short = str(action_args)[:120]
            res_short = json.dumps(result)[:200]
            self.memory.append(f"pos={pos} action={action_name} args={args_short} result={res_short}")
            if len(self.memory) > self.memory_limit:
                self.memory = self.memory[-self.memory_limit:]
            return result

        # Exploration: prefer nearest unscanned in sector
        target_unscanned = self._nearest_unscanned_in_sector()
        if target_unscanned is not None:
            # Occasionally scan to mark area, otherwise move toward target
            if (self.step_count % 4) == 0:
                action_name = "scan"
                action_args = {"radius": 2}
            else:
                direction = self._direction_toward(target_unscanned, pos)
                action_name = "move"
                action_args = {"direction": direction}
        else:
            # No unscanned left in sector: drift toward sector center or broadcast
            if self.sector_bounds is not None:
                x0, x1, y0, y1 = self.sector_bounds
                center = (int((x0 + x1) // 2), int((y0 + y1) // 2), pos[2])
                direction = self._direction_toward(center, pos)
                action_name = "move"
                action_args = {"direction": direction}
            else:
                action_name = "send_message"
                action_args = {"recipient_id": None, "message": "Sector fully covered; returning soon."}
        result = self._execute_action(action_name, action_args)
        # compact log line without embedding prior memory/state
        pos = self.env.get_drone_position(self.drone_id)
        args_short = str(action_args)[:120]
        res_short = json.dumps(result)[:200]
        self.memory.append(f"pos={pos} action={action_name} args={args_short} result={res_short}")
        # cap memory to avoid unbounded growth
        if len(self.memory) > self.memory_limit:
            self.memory = self.memory[-self.memory_limit:]
        return result

    # Execute mapped tool
    def _execute_action(self, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        tool = self.tools.get(name)
        if tool is None:
            return {"ok": False, "error": f"unknown tool {name}"}
        try:
            if name in ("move", "scan"):
                return tool(self.env, self.drone_id, **args)  # type: ignore[arg-type]
            elif name == "move_delta":
                return tool(self.env, self.drone_id, args.get("delta", (0, 0, 0)))
            elif name == "send_message":
                return tool(**args)
            else:
                return {"ok": False, "error": f"unsupported tool {name}"}
        except Exception as e:  # pragma: no cover
            return {"ok": False, "error": str(e)} 