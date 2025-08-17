from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from core.environment import Environment, Vector3


@dataclass
class Message:
    sender_id: str
    recipient_id: str | None  # None means broadcast
    content: str


@dataclass
class SwarmCommand:
    drone_id: str
    action: str  # move, scan, send_message
    args: Dict


@dataclass
class Swarm:
    env: Environment
    drone_ids: List[str]
    inboxes: Dict[str, List[Message]] = field(default_factory=dict)
    broadcast_log: List[Message] = field(default_factory=list)

    def __post_init__(self) -> None:
        for drone_id in self.drone_ids:
            self.env.register_drone(drone_id)
            self.inboxes.setdefault(drone_id, [])

    # Messaging
    def send_message(self, sender_id: str, recipient_id: Optional[str], content: str) -> None:
        msg = Message(sender_id=sender_id, recipient_id=recipient_id, content=content)
        if recipient_id is None:
            self.broadcast_log.append(msg)
            for drone_id in self.drone_ids:
                if drone_id != sender_id:
                    self.inboxes[drone_id].append(msg)
        else:
            if recipient_id in self.inboxes:
                self.inboxes[recipient_id].append(msg)

    def get_messages(self, drone_id: str) -> List[Message]:
        messages = self.inboxes.get(drone_id, [])
        self.inboxes[drone_id] = []
        return messages

    # Centralized execution
    def execute_swarm_commands(self, commands: List[SwarmCommand]) -> List[Dict]:
        results = []
        for cmd in commands:
            if cmd.drone_id not in self.drone_ids:
                results.append({"ok": False, "error": f"unknown drone {cmd.drone_id}"})
                continue
            
            if cmd.action == "move":
                from tools.movement import move
                result = move(self.env, cmd.drone_id, cmd.args.get("direction", "north"))
            elif cmd.action == "scan":
                from tools.movement import scan
                result = scan(self.env, cmd.drone_id, cmd.args.get("radius", 2))
            elif cmd.action == "send_message":
                self.send_message(
                    sender_id=cmd.drone_id,
                    recipient_id=cmd.args.get("recipient_id"),
                    content=cmd.args.get("message", "")
                )
                result = {"ok": True}
            else:
                result = {"ok": False, "error": f"unknown action {cmd.action}"}
            
            results.append(result)
        return results

    # Telemetry
    def telemetry(self) -> Dict:
        return {
            "positions": {d: self.env.get_drone_position(d) for d in self.drone_ids},
            "coverage": float(self.env.scanned.sum()) / float(self.env.scanned.size),
            "discovered_targets": [t.id for t in self.env.discovered_targets()],
            "remaining_targets": [t.id for t in self.env.remaining_targets()],
        }

    def get_all_drone_states(self) -> Dict[str, Dict]:
        """Get the state snapshot for all drones"""
        states = {}
        for drone_id in self.drone_ids:
            pos = self.env.get_drone_position(drone_id)
            messages = self.get_messages(drone_id)
            msgs = [f"from {m.sender_id}: {m.content}" for m in messages]
            
            states[drone_id] = {
                "position": pos,
                "messages": msgs,
            }
        return states 