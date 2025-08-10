from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np


Vector3 = Tuple[int, int, int]


@dataclass
class Target:
    id: str
    position: Vector3
    found: bool = False


@dataclass
class Environment:
    size_x: int = 20
    size_y: int = 20
    size_z: int = 5
    num_targets: int = 5
    rng_seed: int = 42

    # runtime state
    targets: Dict[str, Target] = field(default_factory=dict)
    scanned: np.ndarray | None = None
    drone_positions: Dict[str, Vector3] = field(default_factory=dict)

    def __post_init__(self) -> None:
        rng = np.random.default_rng(self.rng_seed)
        self.scanned = np.zeros((self.size_x, self.size_y, self.size_z), dtype=np.uint8)
        taken: set[Tuple[int, int, int]] = set()
        for t in range(self.num_targets):
            while True:
                pos = (
                    int(rng.integers(0, self.size_x)),
                    int(rng.integers(0, self.size_y)),
                    int(rng.integers(0, self.size_z)),
                )
                if pos not in taken:
                    taken.add(pos)
                    break
            self.targets[f"T{t}"] = Target(id=f"T{t}", position=pos)

    # Bounds and helpers
    def in_bounds(self, p: Vector3) -> bool:
        x, y, z = p
        return 0 <= x < self.size_x and 0 <= y < self.size_y and 0 <= z < self.size_z

    def clamp(self, p: Vector3) -> Vector3:
        x, y, z = p
        return (
            max(0, min(self.size_x - 1, x)),
            max(0, min(self.size_y - 1, y)),
            max(0, min(self.size_z - 1, z)),
        )

    # Drones
    def register_drone(self, drone_id: str, start: Optional[Vector3] = None) -> Vector3:
        if start is None:
            # spawn at ground level center-ish
            start = (
                int(self.size_x // 2),
                int(self.size_y // 2),
                0,
            )
        start = self.clamp(start)
        self.drone_positions[drone_id] = start
        return start

    def get_drone_position(self, drone_id: str) -> Vector3:
        return self.drone_positions[drone_id]

    def move_drone(self, drone_id: str, delta: Vector3) -> Vector3:
        x, y, z = self.drone_positions[drone_id]
        dx, dy, dz = delta
        np.seterr(over="ignore")
        new_pos = (x + int(dx), y + int(dy), z + int(dz))
        new_pos = self.clamp(new_pos)
        self.drone_positions[drone_id] = new_pos
        return new_pos

    # Scanning
    def scan_sphere(self, center: Vector3, radius: int) -> Dict:
        cx, cy, cz = center
        r = max(0, int(radius))
        found_targets: List[str] = []

        x0 = max(0, cx - r)
        x1 = min(self.size_x - 1, cx + r)
        y0 = max(0, cy - r)
        y1 = min(self.size_y - 1, cy + r)
        z0 = max(0, cz - r)
        z1 = min(self.size_z - 1, cz + r)

        rr = r * r
        for x in range(x0, x1 + 1):
            for y in range(y0, y1 + 1):
                for z in range(z0, z1 + 1):
                    if (x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2 <= rr:
                        self.scanned[x, y, z] = 1
        for tid, tgt in self.targets.items():
            if not tgt.found:
                tx, ty, tz = tgt.position
                if (tx - cx) ** 2 + (ty - cy) ** 2 + (tz - cz) ** 2 <= rr:
                    tgt.found = True
                    found_targets.append(tid)

        coverage = float(self.scanned.sum()) / float(self.scanned.size)
        return {
            "found_targets": found_targets,
            "coverage": coverage,
        }

    # Telemetry helpers
    def discovered_targets(self) -> List[Target]:
        return [t for t in self.targets.values() if t.found]

    def remaining_targets(self) -> List[Target]:
        return [t for t in self.targets.values() if not t.found] 