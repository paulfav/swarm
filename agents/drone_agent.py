from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np

from core.environment import Environment
from core.swarm import Swarm
from tools import movement

try:
    from openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore


def get_llm_provider() -> str:
    return os.getenv("LLM_PROVIDER", "mock").lower()


@dataclass
class PlannedAction:
    action: str  # move, scan, send_message
    args: Dict[str, Any]
    justification: str


@dataclass
class DroneAgent:
    drone_id: str
    env: Environment
    swarm: Swarm
    tools: Dict[str, Callable] = field(default_factory=dict)
    memory: List[str] = field(default_factory=list)
    memory_limit: int = 100
    logger: Optional[Any] = None  # RunLogger instance
    
    # Planning system
    action_plan: List[PlannedAction] = field(default_factory=list)
    plan_step: int = 0
    step_count: int = 0
    last_llm_call_step: int = 0
    
    # Sector and exploration
    sector_bounds: Tuple[int, int, int, int] | None = None
    home_position: Tuple[int, int, int] | None = None

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

        # Initialize sector and home
        self.home_position = self.env.get_drone_position(self.drone_id)
        self.sector_bounds = self._compute_sector_bounds()

    def _compute_sector_bounds(self) -> Tuple[int, int, int, int]:
        """Compute sector bounds for this drone"""
        sx, sy = self.env.size_x, self.env.size_y
        ids = self.swarm.drone_ids
        n = max(1, len(ids))
        idx = max(0, ids.index(self.drone_id)) if self.drone_id in ids else 0
        slice_width = sx // n
        x0 = idx * slice_width
        x1 = (idx + 1) * slice_width - 1 if idx < n - 1 else sx - 1
        y0, y1 = 0, sy - 1
        return (x0, x1, y0, y1)

    def _nearest_unscanned_in_sector(self) -> Optional[Tuple[int, int, int]]:
        """Find nearest unscanned location in drone's sector"""
        x0, x1, y0, y1 = self.sector_bounds
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
        distances = np.abs(xs - cx) + np.abs(ys - cy) + (dz * 2)
        i = int(np.argmin(distances))
        return (int(xs[i]), int(ys[i]), int(zs[i]))

    def _coverage_in_sector(self) -> float:
        """Get coverage percentage in drone's sector"""
        x0, x1, y0, y1 = self.sector_bounds
        scanned_slice = self.env.scanned[x0:x1 + 1, y0:y1 + 1, :]
        return float(scanned_slice.sum()) / float(scanned_slice.size)

    def _direction_toward(self, target_pos: Tuple[int, int, int], current: Tuple[int, int, int]) -> str:
        """Get direction to move toward target"""
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

    def _is_wall_encountered(self, direction: str) -> bool:
        """Check if moving in direction would hit a wall"""
        pos = self.env.get_drone_position(self.drone_id)
        from tools.movement import DIRECTION_TO_DELTA
        if direction in DIRECTION_TO_DELTA:
            delta = DIRECTION_TO_DELTA[direction]
            new_pos = (
                pos[0] + delta[0],
                pos[1] + delta[1],
                pos[2] + delta[2]
            )
            return not self.env.in_bounds(new_pos)
        return False

    def _is_obstacle_found(self) -> bool:
        """Check if drone has found something interesting (target, high unscanned density)"""
        pos = self.env.get_drone_position(self.drone_id)
        
        # Check if there are many unscanned cells nearby
        x, y, z = pos
        unscanned_count = 0
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                for dz in range(-1, 2):
                    nx, ny, nz = x + dx, y + dy, z + dz
                    if (0 <= nx < self.env.size_x and 
                        0 <= ny < self.env.size_y and 
                        0 <= nz < self.env.size_z and
                        self.env.scanned[nx, ny, nz] == 0):
                        unscanned_count += 1
        
        return unscanned_count > 10

    def _build_state_snapshot(self) -> str:
        """Build comprehensive state for LLM planning"""
        pos = self.env.get_drone_position(self.drone_id)
        remaining = [t.id for t in self.env.remaining_targets()]
        discovered = [t.id for t in self.env.discovered_targets()]
        inbox = self.swarm.get_messages(self.drone_id)
        msgs = [f"from {m.sender_id}: {m.content}" for m in inbox]
        coverage = float(self.env.scanned.sum()) / float(self.env.scanned.size)
        sector_cov = self._coverage_in_sector()
        nearest_unscanned = self._nearest_unscanned_in_sector()
        
        # Get other drone positions for coordination
        other_drones = {}
        for drone_id in self.swarm.drone_ids:
            if drone_id != self.drone_id:
                other_drones[drone_id] = self.env.get_drone_position(drone_id)

        return json.dumps({
            "drone_id": self.drone_id,
            "position": pos,
            "sector_bounds": self.sector_bounds,
            "sector_coverage": sector_cov,
            "nearest_unscanned_in_sector": nearest_unscanned,
            "overall_coverage": coverage,
            "remaining_targets": remaining,
            "discovered_targets": discovered,
            "messages": msgs,
            "other_drones": other_drones,
            "step_count": self.step_count,
            "plan_step": self.plan_step,
        })

    def _build_system_prompt(self) -> str:
        return (
            f"You are an autonomous search-and-rescue drone ({self.drone_id}) with your own LLM brain. "
            "You plan 10 steps ahead to maximize efficiency and only replan when necessary.\n\n"
            "MISSION: Find all targets as quickly as possible in your assigned sector.\n\n"
            "PLANNING STRATEGY:\n"
            "1. Create a 10-step plan that maximizes coverage in your sector\n"
            "2. Use systematic grid-like movement patterns\n"
            "3. Scan when in areas with high unscanned density\n"
            "4. Coordinate with other drones through messages\n"
            "5. Stay within your sector boundaries\n\n"
            "AVAILABLE ACTIONS:\n"
            "- move(direction): north, south, east, west, up, down, northeast, northwest, southeast, southwest\n"
            "- scan(radius): radius 1-4, reveals targets and marks area\n"
            "- send_message(recipient_id, message): communicate with other drones\n\n"
            "REQUIREMENTS:\n"
            "- Provide exactly 10 planned actions\n"
            "- Each action must include justification\n"
            "- Plan should be efficient and systematic\n"
            "- Consider sector boundaries and other drones\n"
            "- Prioritize unexplored areas\n\n"
            "EFFICIENCY TIPS:\n"
            "- Use diagonal movements to cover more ground\n"
            "- Scan when many unscanned cells nearby\n"
            "- Move toward unscanned areas\n"
            "- Coordinate with other drones to avoid overlap"
        )

    def _should_call_llm(self) -> bool:
        """Determine if LLM should be called for new planning"""
        # Call every 10 steps
        if self.step_count - self.last_llm_call_step >= 10:
            return True
        
        # Call if wall encountered
        if self.action_plan and self.plan_step < len(self.action_plan):
            next_action = self.action_plan[self.plan_step]
            if next_action.action == "move":
                direction = next_action.args.get("direction", "north")
                if self._is_wall_encountered(direction):
                    return True
        
        # Call if obstacle/interesting area found
        if self._is_obstacle_found():
            return True
        
        return False

    def _call_llm_for_planning(self) -> List[PlannedAction]:
        """Call LLM to create a new 10-step plan"""
        state = self._build_state_snapshot()
        system = self._build_system_prompt()

        if get_llm_provider() == "openai" and self._client is not None:
            return self._openai_plan(system, state)
        else:
            return self._mock_plan(system, state)

    def _openai_plan(self, system_prompt: str, state_json: str) -> List[PlannedAction]:
        """Use OpenAI to create a 10-step plan"""
        tool_defs = [
            {
                "type": "function",
                "function": {
                    "name": "plan_action",
                    "description": "Plan a single action for the drone",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "action": {"type": "string", "enum": ["move", "scan", "send_message"]},
                            "direction": {"type": "string", "description": "For move action"},
                            "radius": {"type": "integer", "description": "For scan action"},
                            "recipient_id": {"type": ["string", "null"], "description": "For send_message action"},
                            "message": {"type": "string", "description": "For send_message action"},
                            "justification": {"type": "string", "description": "Explain why this action was chosen"}
                        },
                        "required": ["action", "justification"]
                    }
                }
            }
        ]

        user_input = f"DRONE STATE: {state_json}\nCreate a 10-step plan for efficient exploration. Provide exactly 10 planned actions with justifications."
        
        resp = self._client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            temperature=0.2,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input},
            ],
            tools=tool_defs,
            tool_choice="auto",
        )
        choice = resp.choices[0]
        tool_calls = getattr(choice.message, "tool_calls", None) or []
        
        # Log the planning conversation
        if self.logger:
            self.logger.log_llm_conversation(
                drone_id=self.drone_id,
                step=self.step_count,
                system_prompt=system_prompt,
                user_input=user_input,
                response={
                    "model": resp.model,
                    "usage": resp.usage.model_dump() if resp.usage else None,
                    "finish_reason": choice.finish_reason,
                    "content": choice.message.content,
                },
                tool_calls=[tc.model_dump() for tc in tool_calls]
            )
        
        # Convert tool calls to planned actions
        planned_actions = []
        for call in tool_calls:
            args = json.loads(call.function.arguments or "{}")
            action = args.get("action")
            justification = args.get("justification", "No justification")
            
            if not action:
                continue
                
            # Build action args
            action_args = {}
            if action == "move":
                action_args["direction"] = args.get("direction", "north")
            elif action == "scan":
                action_args["radius"] = args.get("radius", 2)
            elif action == "send_message":
                action_args["recipient_id"] = args.get("recipient_id")
                action_args["message"] = args.get("message", "")
            
            planned_actions.append(PlannedAction(
                action=action,
                args=action_args,
                justification=justification
            ))
        
        # Ensure we have exactly 10 actions (pad with moves if needed)
        while len(planned_actions) < 10:
            planned_actions.append(PlannedAction(
                action="move",
                args={"direction": "north"},
                justification="Default move to complete 10-step plan"
            ))
        
        return planned_actions[:10]  # Ensure exactly 10

    def _mock_plan(self, system_prompt: str, state_json: str) -> List[PlannedAction]:
        """Mock planning for when OpenAI is not available"""
        planned_actions = []
        rnd = random.Random(f"{self.drone_id}_{self.step_count}")
        
        for i in range(10):
            pos = self.env.get_drone_position(self.drone_id)
            nearest_unscanned = self._nearest_unscanned_in_sector()
            
            if nearest_unscanned and rnd.random() < 0.3:
                # Scan action
                planned_actions.append(PlannedAction(
                    action="scan",
                    args={"radius": 2},
                    justification=f"Step {i+1}: Scanning to reveal unscanned area at {nearest_unscanned}"
                ))
            else:
                # Move action
                if nearest_unscanned:
                    direction = self._direction_toward(nearest_unscanned, pos)
                else:
                    # Move toward sector center
                    x0, x1, y0, y1 = self.sector_bounds
                    center = (int((x0 + x1) // 2), int((y0 + y1) // 2), pos[2])
                    direction = self._direction_toward(center, pos)
                
                planned_actions.append(PlannedAction(
                    action="move",
                    args={"direction": direction},
                    justification=f"Step {i+1}: Moving {direction} toward optimal exploration direction"
                ))
        
        return planned_actions

    def step(self) -> Dict[str, Any]:
        """Execute one step for this drone"""
        self.step_count += 1
        
        # Check if we need to call LLM for new planning
        if self._should_call_llm():
            self.action_plan = self._call_llm_for_planning()
            self.plan_step = 0
            self.last_llm_call_step = self.step_count
        
        # Execute next action from plan
        if self.action_plan and self.plan_step < len(self.action_plan):
            planned_action = self.action_plan[self.plan_step]
            result = self._execute_action(planned_action.action, planned_action.args)
            
            # Log the action
            pos = self.env.get_drone_position(self.drone_id)
            args_short = str(planned_action.args)[:120]
            res_short = json.dumps(result)[:200]
            self.memory.append(f"step={self.step_count} plan_step={self.plan_step} action={planned_action.action} args={args_short} justification={planned_action.justification} result={res_short}")
            
            self.plan_step += 1
        else:
            # Fallback action if no plan
            result = self._execute_action("move", {"direction": "north"})
            self.memory.append(f"step={self.step_count} fallback_action=move result={json.dumps(result)}")
        
        # Cap memory
        if len(self.memory) > self.memory_limit:
            self.memory = self.memory[-self.memory_limit:]
        
        return result

    def _execute_action(self, action: str, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a planned action"""
        tool = self.tools.get(action)
        if tool is None:
            return {"ok": False, "error": f"unknown action {action}"}
        
        try:
            if action in ("move", "scan"):
                return tool(self.env, self.drone_id, **args)
            elif action == "move_delta":
                return tool(self.env, self.drone_id, args.get("delta", (0, 0, 0)))
            elif action == "send_message":
                return tool(**args)
            else:
                return {"ok": False, "error": f"unsupported action {action}"}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def _tool_send_message(self, recipient_id: Optional[str] = None, message: str = "") -> Dict[str, Any]:
        """Tool wrapper for sending messages"""
        self.swarm.send_message(sender_id=self.drone_id, recipient_id=recipient_id, content=message)
        return {"ok": True} 