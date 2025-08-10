from __future__ import annotations

from typing import Dict, Tuple
from core.environment import Environment, Vector3


DIRECTION_TO_DELTA: Dict[str, Vector3] = {
    "up": (0, 0, 1),
    "down": (0, 0, -1),
    "north": (0, 1, 0),
    "south": (0, -1, 0),
    "east": (1, 0, 0),
    "west": (-1, 0, 0),
    # diagonals (optional)
    "northeast": (1, 1, 0),
    "northwest": (-1, 1, 0),
    "southeast": (1, -1, 0),
    "southwest": (-1, -1, 0),
}


def move(env: Environment, drone_id: str, direction: str) -> Dict:
    direction_key = direction.strip().lower()
    if direction_key not in DIRECTION_TO_DELTA:
        return {"ok": False, "error": f"unknown direction '{direction}'"}
    new_pos = env.move_drone(drone_id, DIRECTION_TO_DELTA[direction_key])
    return {"ok": True, "position": new_pos}


def move_delta(env: Environment, drone_id: str, delta: Tuple[int, int, int]) -> Dict:
    new_pos = env.move_drone(drone_id, delta)
    return {"ok": True, "position": new_pos}


def scan(env: Environment, drone_id: str, radius: int) -> Dict:
    center = env.get_drone_position(drone_id)
    result = env.scan_sphere(center, radius)
    return {"ok": True, **result} 