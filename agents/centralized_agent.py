from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from core.environment import Environment
from core.swarm import Swarm, SwarmCommand
from tools import movement

try:
    from openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore


def get_llm_provider() -> str:
    return os.getenv("LLM_PROVIDER", "mock").lower()


@dataclass
class CentralizedAgent:
    env: Environment
    swarm: Swarm
    logger: Optional[Any] = None  # RunLogger instance
    step_count: int = 0
    memory: List[str] = field(default_factory=list)
    memory_limit: int = 100
    initial_spread_done: bool = False
    drone_directions: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        provider = get_llm_provider()
        if provider == "openai" and OpenAI is not None:
            self._client = OpenAI()
        else:
            self._client = None

    def _compute_sector_bounds(self, drone_id: str) -> Tuple[int, int, int, int]:
        # Divide X range into equal vertical slices by drone index
        sx, sy = self.env.size_x, self.env.size_y
        ids = self.swarm.drone_ids
        n = max(1, len(ids))
        idx = max(0, ids.index(drone_id)) if drone_id in ids else 0
        slice_width = sx // n
        x0 = idx * slice_width
        x1 = (idx + 1) * slice_width - 1 if idx < n - 1 else sx - 1
        y0, y1 = 0, sy - 1
        return (x0, x1, y0, y1)

    def _nearest_unscanned_in_sector(self, drone_id: str) -> Optional[Tuple[int, int, int]]:
        sector_bounds = self._compute_sector_bounds(drone_id)
        x0, x1, y0, y1 = sector_bounds
        scanned = self.env.scanned
        xs, ys, zs = np.where(scanned == 0)
        # mask to sector bounds
        mask = (xs >= x0) & (xs <= x1) & (ys >= y0) & (ys <= y1)
        xs, ys, zs = xs[mask], ys[mask], zs[mask]
        if xs.size == 0:
            return None
        cx, cy, cz = self.env.get_drone_position(drone_id)
        # prioritize same Z, then closest by L1 distance
        dz = np.abs(zs - cz)
        # weight z difference slightly higher to favor same altitude
        distances = np.abs(xs - cx) + np.abs(ys - cy) + (dz * 2)
        i = int(np.argmin(distances))
        return (int(xs[i]), int(ys[i]), int(zs[i]))

    def _coverage_in_sector(self, drone_id: str) -> float:
        sector_bounds = self._compute_sector_bounds(drone_id)
        x0, x1, y0, y1 = sector_bounds
        scanned_slice = self.env.scanned[x0:x1 + 1, y0:y1 + 1, :]
        return float(scanned_slice.sum()) / float(scanned_slice.size)

    def _build_swarm_state_snapshot(self) -> str:
        """Build a comprehensive state snapshot for the entire swarm"""
        drone_states = self.swarm.get_all_drone_states()
        remaining = [t.id for t in self.env.remaining_targets()]
        discovered = [t.id for t in self.env.discovered_targets()]
        coverage = float(self.env.scanned.sum()) / float(self.env.scanned.size)
        
        # Enrich each drone's state with sector info
        for drone_id in self.swarm.drone_ids:
            pos = drone_states[drone_id]["position"]
            sector_bounds = self._compute_sector_bounds(drone_id)
            sector_cov = self._coverage_in_sector(drone_id)
            nearest_unscanned = self._nearest_unscanned_in_sector(drone_id)
            
            drone_states[drone_id].update({
                "sector_bounds": sector_bounds,
                "sector_coverage": sector_cov,
                "nearest_unscanned_in_sector": nearest_unscanned,
            })

        return json.dumps({
            "step": self.step_count,
            "overall_coverage": coverage,
            "remaining_targets": remaining,
            "discovered_targets": discovered,
            "drones": drone_states,
        })

    def _compute_initial_spread_directions(self) -> Dict[str, str]:
        """Compute optimal initial directions for each drone to spread out"""
        n_drones = len(self.swarm.drone_ids)
        if n_drones == 1:
            return {self.swarm.drone_ids[0]: "northeast"}
        
        # Define spread directions based on number of drones
        if n_drones == 2:
            directions = ["northwest", "southeast"]
        elif n_drones == 3:
            directions = ["northwest", "northeast", "south"]
        elif n_drones == 4:
            directions = ["northwest", "northeast", "southwest", "southeast"]
        elif n_drones == 5:
            directions = ["northwest", "north", "northeast", "southwest", "southeast"]
        elif n_drones == 6:
            directions = ["northwest", "north", "northeast", "southwest", "south", "southeast"]
        else:
            # For more drones, use a more complex pattern
            base_directions = ["northwest", "north", "northeast", "east", "southeast", "south", "southwest", "west"]
            directions = base_directions[:n_drones]
        
        return {drone_id: directions[i] for i, drone_id in enumerate(self.swarm.drone_ids)}

    def _get_optimal_direction_for_drone(self, drone_id: str) -> str:
        """Get the optimal direction for a drone based on its sector and current position"""
        pos = self.env.get_drone_position(drone_id)
        sector_bounds = self._compute_sector_bounds(drone_id)
        x0, x1, y0, y1 = sector_bounds
        
        # If we haven't done initial spread, use pre-computed directions
        if not self.initial_spread_done:
            return self.drone_directions.get(drone_id, "north")
        
        # Find nearest unscanned in sector
        nearest_unscanned = self._nearest_unscanned_in_sector(drone_id)
        if nearest_unscanned:
            return self._direction_toward(nearest_unscanned, pos)
        
        # If no unscanned in sector, move toward sector center
        center = (int((x0 + x1) // 2), int((y0 + y1) // 2), pos[2])
        return self._direction_toward(center, pos)

    def _should_scan_now(self, drone_id: str) -> bool:
        """Determine if a drone should scan based on its environment"""
        pos = self.env.get_drone_position(drone_id)
        
        # Count unscanned cells in immediate vicinity
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
        
        # Scan if there are many unscanned cells nearby or occasionally
        return unscanned_count > 8 or (self.step_count % 5 == 0)

    def _build_system_prompt(self) -> str:
        return (
            "You are a centralized AI controller for a search-and-rescue drone swarm. "
            "Your mission is to find all targets as quickly as possible using coordinated search patterns.\n\n"
            "SEARCH STRATEGY:\n"
            "1. INITIAL SPREAD: Drones start from center and move in different directions to cover the map efficiently\n"
            "2. SECTOR EXPLORATION: Each drone explores its assigned sector (vertical slice of the map)\n"
            "3. SYSTEMATIC SCANNING: Scan in a grid pattern, moving methodically through each sector\n"
            "4. TARGET FOCUS: When a target is discovered, nearby drones should converge to scan the area thoroughly\n"
            "5. COVERAGE OPTIMIZATION: Prioritize areas with high unscanned density\n"
            "6. PROXIMITY CONSTRAINT: Keep all drones within radius 8 of each other for coordination\n"
            "7. DIRECTIONAL COORDINATION: Ensure drones move in complementary directions to avoid overlap\n\n"
            "AVAILABLE ACTIONS per drone:\n"
            "- move(direction): north, south, east, west, up, down, northeast, northwest, southeast, southwest\n"
            "- scan(radius): radius 1-4, reveals targets and marks area as scanned\n"
            "- send_message(recipient_id, message): communicate with other drones\n\n"
            "REQUIREMENTS:\n"
            "- Provide exactly one action per drone per step\n"
            "- Include written justification for each action\n"
            "- Every 10 steps, provide a strategy report explaining your approach and next 10-step plan\n"
            "- Always justify why you chose each action and how it advances the search\n"
            "- Coordinate drones to avoid redundant coverage\n"
            "- When targets are found, focus scanning around that area\n"
            "- Maintain swarm cohesion within radius 8\n"
            "- Ensure drones move in different directions initially to spread out quickly\n\n"
            "EFFICIENCY TIPS:\n"
            "- Use diagonal movements to cover more ground quickly\n"
            "- Scan when in areas with many unscanned cells nearby\n"
            "- Move toward unscanned areas rather than already-scanned regions\n"
            "- Coordinate so drones don't scan the same areas\n"
            "- Prioritize sectors with lower coverage percentages\n"
            "- Start with aggressive spreading, then focus on systematic coverage"
        )

    def _check_proximity_constraint(self, commands: List[SwarmCommand]) -> bool:
        """Check if proposed moves would keep drones within radius R of each other"""
        R = 8  # proximity radius
        current_positions = {d: self.env.get_drone_position(d) for d in self.swarm.drone_ids}
        
        # Simulate the moves
        proposed_positions = current_positions.copy()
        for cmd in commands:
            if cmd.action == "move":
                from tools.movement import DIRECTION_TO_DELTA
                direction = cmd.args.get("direction", "north")
                if direction in DIRECTION_TO_DELTA:
                    delta = DIRECTION_TO_DELTA[direction]
                    current = proposed_positions[cmd.drone_id]
                    new_pos = (
                        max(0, min(self.env.size_x - 1, current[0] + delta[0])),
                        max(0, min(self.env.size_y - 1, current[1] + delta[1])),
                        max(0, min(self.env.size_z - 1, current[2] + delta[2]))
                    )
                    proposed_positions[cmd.drone_id] = new_pos
        
        # Check if all drones are within radius R of each other
        positions = list(proposed_positions.values())
        for i, pos1 in enumerate(positions):
            for j, pos2 in enumerate(positions[i+1:], i+1):
                distance = abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1]) + abs(pos1[2] - pos2[2])
                if distance > R:
                    return False
        return True

    def _generate_strategy_report(self) -> str:
        """Generate a comprehensive strategy report every 10 steps"""
        coverage = float(self.env.scanned.sum()) / float(self.env.scanned.size)
        remaining = len(self.env.remaining_targets())
        discovered = len(self.env.discovered_targets())
        
        # Analyze sector coverage
        sector_analysis = {}
        for drone_id in self.swarm.drone_ids:
            sector_cov = self._coverage_in_sector(drone_id)
            sector_analysis[drone_id] = sector_cov
        
        # Find areas with high unscanned density
        scanned = self.env.scanned
        xs, ys, zs = np.where(scanned == 0)
        if xs.size > 0:
            # Find clusters of unscanned areas
            from collections import defaultdict
            clusters = defaultdict(int)
            for x, y, z in zip(xs, ys, zs):
                cluster_key = (x // 4, y // 4, z // 2)  # Group into 4x4x2 clusters
                clusters[cluster_key] += 1
            
            # Top 3 unscanned clusters
            top_clusters = sorted(clusters.items(), key=lambda x: x[1], reverse=True)[:3]
        else:
            top_clusters = []
        
        report = f"""
=== STRATEGY REPORT (Step {self.step_count}) ===
OVERALL STATUS:
- Coverage: {coverage:.1%}
- Targets Found: {discovered}/{discovered + remaining}
- Remaining Targets: {remaining}

SECTOR ANALYSIS:
"""
        for drone_id, cov in sector_analysis.items():
            report += f"- {drone_id}: {cov:.1%} coverage\n"
        
        if top_clusters:
            report += "\nHIGH PRIORITY UNSCANNED AREAS:\n"
            for (cx, cy, cz), count in top_clusters:
                x_range = f"[{cx*4}-{(cx+1)*4-1}]"
                y_range = f"[{cy*4}-{(cy+1)*4-1}]"
                z_range = f"[{cz*2}-{(cz+1)*2-1}]"
                report += f"- Region {x_range}x{y_range}x{z_range}: {count} unscanned cells\n"
        
        report += f"""
NEXT 10-STEP STRATEGY:
1. Focus on sectors with lowest coverage: {min(sector_analysis.items(), key=lambda x: x[1])[0]}
2. Target high-density unscanned clusters
3. Maintain swarm cohesion (radius 8)
4. Systematic grid scanning in priority areas
5. Coordinate to avoid redundant coverage

DRONE ASSIGNMENTS:
"""
        for drone_id in self.swarm.drone_ids:
            pos = self.env.get_drone_position(drone_id)
            sector_cov = sector_analysis[drone_id]
            report += f"- {drone_id} at {pos}: {sector_cov:.1%} sector coverage\n"
        
        return report

    def step(self) -> List[Dict[str, Any]]:
        """Execute one step for the entire swarm"""
        self.step_count += 1
        state = self._build_swarm_state_snapshot()
        system = self._build_system_prompt()

        if get_llm_provider() == "openai" and self._client is not None:
            return self._openai_decide_and_act(system, state)
        else:
            return self._mock_decide_and_act(system, state)

    def _openai_decide_and_act(self, system_prompt: str, state_json: str) -> List[Dict[str, Any]]:
        tool_defs = [
            {
                "type": "function",
                "function": {
                    "name": "swarm_action",
                    "description": "Execute an action for a specific drone in the swarm",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "drone_id": {"type": "string"},
                            "action": {"type": "string", "enum": ["move", "scan", "send_message"]},
                            "direction": {"type": "string", "description": "For move action"},
                            "radius": {"type": "integer", "description": "For scan action"},
                            "recipient_id": {"type": ["string", "null"], "description": "For send_message action"},
                            "message": {"type": "string", "description": "For send_message action"},
                            "justification": {"type": "string", "description": "Explain why this action was chosen"}
                        },
                        "required": ["drone_id", "action", "justification"]
                    }
                }
            }
        ]

        # Add initial spread information to the prompt
        spread_info = ""
        if not self.initial_spread_done:
            spread_info = "\n\nINITIAL SPREAD PHASE: Drones should move in different directions to spread out quickly. Recommended directions: " + ", ".join([f"{d}: {dir}" for d, dir in self.drone_directions.items()])

        # Add strategy report request every 10 steps
        report_request = ""
        if self.step_count % 10 == 0:
            report_request = "\n\nSTRATEGY REPORT REQUEST: Provide a detailed strategy report explaining your current approach, progress analysis, and plan for the next 10 steps. Include coverage analysis, target discovery status, and coordination strategy."

        user_input = f"SWARM STATE: {state_json}\nProvide exactly one action for each drone in the swarm. Include justification for each action.{spread_info}{report_request}"
        
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
        
        # Log the full conversation
        if self.logger:
            self.logger.log_llm_conversation(
                drone_id="CENTRAL",
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
        
        # Convert tool calls to swarm commands
        commands = []
        justifications = {}
        for call in tool_calls:
            args = json.loads(call.function.arguments or "{}")
            drone_id = args.get("drone_id")
            action = args.get("action")
            justification = args.get("justification", "No justification provided")
            
            if not drone_id or not action:
                continue
                
            # Build command args based on action
            cmd_args = {}
            if action == "move":
                cmd_args["direction"] = args.get("direction", "north")
            elif action == "scan":
                cmd_args["radius"] = args.get("radius", 2)
            elif action == "send_message":
                cmd_args["recipient_id"] = args.get("recipient_id")
                cmd_args["message"] = args.get("message", "")
            
            commands.append(SwarmCommand(drone_id=drone_id, action=action, args=cmd_args))
            justifications[drone_id] = justification
        
        # Check proximity constraint
        if not self._check_proximity_constraint(commands):
            # If constraint violated, modify commands to keep drones closer
            commands = self._adjust_commands_for_proximity(commands)
        
        # Execute all commands
        results = self.swarm.execute_swarm_commands(commands)
        
        # Mark initial spread as done after a few steps
        if self.step_count >= 5:
            self.initial_spread_done = True
        
        # Log results with justifications
        for i, result in enumerate(results):
            if i < len(commands):
                cmd = commands[i]
                pos = self.env.get_drone_position(cmd.drone_id)
                args_short = str(cmd.args)[:120]
                res_short = json.dumps(result)[:200]
                justification = justifications.get(cmd.drone_id, "No justification")
                self.memory.append(f"drone={cmd.drone_id} pos={pos} action={cmd.action} args={args_short} justification={justification} result={res_short}")
        
        # Generate and log strategy report every 10 steps
        if self.step_count % 10 == 0:
            strategy_report = self._generate_strategy_report()
            if self.logger:
                self.logger.log_strategy_report(strategy_report)
        
        if len(self.memory) > self.memory_limit:
            self.memory = self.memory[-self.memory_limit:]
        
        return results

    def _mock_decide_and_act(self, system_prompt: str, state_json: str) -> List[Dict[str, Any]]:
        """Mock centralized decision making with smart spreading"""
        commands = []
        justifications = {}
        rnd = random.Random(f"central_{self.step_count}")
        
        # Initialize spread directions if not done
        if not self.initial_spread_done:
            self.drone_directions = self._compute_initial_spread_directions()
        
        for drone_id in self.swarm.drone_ids:
            pos = self.env.get_drone_position(drone_id)
            
            # Determine action based on phase
            if not self.initial_spread_done and self.step_count < 5:
                # Initial spread phase
                direction = self.drone_directions.get(drone_id, "north")
                commands.append(SwarmCommand(
                    drone_id=drone_id,
                    action="move",
                    args={"direction": direction}
                ))
                justifications[drone_id] = f"Initial spread: moving {direction} to cover assigned direction"
            else:
                # Normal exploration phase
                if self._should_scan_now(drone_id):
                    commands.append(SwarmCommand(
                        drone_id=drone_id,
                        action="scan",
                        args={"radius": 2}
                    ))
                    justifications[drone_id] = f"Scanning at {pos} to reveal nearby unscanned areas"
                else:
                    direction = self._get_optimal_direction_for_drone(drone_id)
                    commands.append(SwarmCommand(
                        drone_id=drone_id,
                        action="move",
                        args={"direction": direction}
                    ))
                    justifications[drone_id] = f"Moving {direction} toward optimal exploration direction"
        
        # Mark initial spread as done
        if self.step_count >= 5:
            self.initial_spread_done = True
        
        # Check proximity constraint
        if not self._check_proximity_constraint(commands):
            commands = self._adjust_commands_for_proximity(commands)
        
        results = self.swarm.execute_swarm_commands(commands)
        
        # Log results with justifications
        for i, result in enumerate(results):
            if i < len(commands):
                cmd = commands[i]
                pos = self.env.get_drone_position(cmd.drone_id)
                args_short = str(cmd.args)[:120]
                res_short = json.dumps(result)[:200]
                justification = justifications.get(cmd.drone_id, "No justification")
                self.memory.append(f"drone={cmd.drone_id} pos={pos} action={cmd.action} args={args_short} justification={justification} result={res_short}")
        
        # Generate and log strategy report every 10 steps
        if self.step_count % 10 == 0:
            strategy_report = self._generate_strategy_report()
            if self.logger:
                self.logger.log_strategy_report(strategy_report)
        
        if len(self.memory) > self.memory_limit:
            self.memory = self.memory[-self.memory_limit:]
        
        return results

    def _direction_toward(self, target_pos: Tuple[int, int, int], current: Tuple[int, int, int]) -> str:
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

    def _adjust_commands_for_proximity(self, commands: List[SwarmCommand]) -> List[SwarmCommand]:
        """Adjust commands to maintain proximity constraint"""
        R = 8
        current_positions = {d: self.env.get_drone_position(d) for d in self.swarm.drone_ids}
        
        # Find the center of the swarm
        center_x = sum(pos[0] for pos in current_positions.values()) / len(current_positions)
        center_y = sum(pos[1] for pos in current_positions.values()) / len(current_positions)
        center_z = sum(pos[2] for pos in current_positions.values()) / len(current_positions)
        center = (int(center_x), int(center_y), int(center_z))
        
        # Adjust commands to move toward center if too far
        adjusted_commands = []
        for cmd in commands:
            if cmd.action == "move":
                current_pos = current_positions[cmd.drone_id]
                distance_from_center = abs(current_pos[0] - center[0]) + abs(current_pos[1] - center[1]) + abs(current_pos[2] - center[2])
                
                if distance_from_center > R // 2:
                    # Move toward center
                    direction = self._direction_toward(center, current_pos)
                    adjusted_commands.append(SwarmCommand(
                        drone_id=cmd.drone_id,
                        action="move",
                        args={"direction": direction}
                    ))
                else:
                    # Original command is fine
                    adjusted_commands.append(cmd)
            else:
                # Non-move commands are fine
                adjusted_commands.append(cmd)
        
        return adjusted_commands
