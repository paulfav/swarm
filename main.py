
from __future__ import annotations

import os
import time
from typing import List

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from matplotlib.animation import FuncAnimation
import gc

from core.environment import Environment
from core.swarm import Swarm
from agents.drone_agent import DroneAgent
from core.config import load_config
from core.logger import RunLogger


def create_swarm_from_config():
    cfg = load_config()
    env = Environment(
        size_x=cfg.environment.size_x,
        size_y=cfg.environment.size_y,
        size_z=cfg.environment.size_z,
        num_targets=cfg.environment.num_targets,
        rng_seed=cfg.environment.rng_seed,
    )
    drone_ids = [f"D{i}" for i in range(cfg.swarm.num_drones)]
    swarm = Swarm(env=env, drone_ids=drone_ids)
    agents = [DroneAgent(drone_id=d, env=env, swarm=swarm, memory_limit=cfg.agent.memory_limit) for d in drone_ids]
    return cfg, env, swarm, agents


class Visualization:
    def __init__(self, env: Environment, swarm: Swarm, agents: List[DroneAgent], max_scanned_points: int = 3000, render_scanned: bool = True, render_targets: bool = True):
        self.env = env
        self.swarm = swarm
        self.agents = agents
        self.max_scanned_points = max_scanned_points
        self.render_scanned = render_scanned
        self.render_targets = render_targets
        self.fig = plt.figure(figsize=(8, 6))
        self.ax = self.fig.add_subplot(111, projection="3d")
        self.ax.set_xlim(0, env.size_x)
        self.ax.set_ylim(0, env.size_y)
        self.ax.set_zlim(0, env.size_z)
        self.ax.set_title("LLM-Driven Drone Swarm Search & Rescue")

        # precompute target locations
        self.target_positions = np.array([t.position for t in env.targets.values()])
        # scatter handles
        self.drone_scatter = None
        self.target_scatter = None

    def init_draw(self):
        self.ax.clear()
        self.ax.set_xlim(0, self.env.size_x)
        self.ax.set_ylim(0, self.env.size_y)
        self.ax.set_zlim(0, self.env.size_z)
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_zlabel("Z")
        self.ax.set_title("LLM-Driven Drone Swarm Search & Rescue")

        # draw scanned voxels as faint points for performance
        if self.render_scanned:
            xs, ys, zs = np.where(self.env.scanned == 1)
            self.scanned_scatter = self.ax.scatter(xs + 0.5, ys + 0.5, zs + 0.5, c="lightblue", alpha=0.2, s=4)
        else:
            self.scanned_scatter = None

        # draw targets
        if self.render_targets and len(self.env.targets) > 0:
            discovered_mask = np.array([t.found for t in self.env.targets.values()])
            colors = np.where(discovered_mask, "green", "red")
            tp = np.array([t.position for t in self.env.targets.values()])
            self.target_scatter = self.ax.scatter(tp[:, 0] + 0.5, tp[:, 1] + 0.5, tp[:, 2] + 0.5, c=colors, s=40, marker="^", label="Targets")
        else:
            self.target_scatter = None

        # draw drones
        positions = np.array([self.env.get_drone_position(a.drone_id) for a in self.agents])
        self.drone_scatter = self.ax.scatter(positions[:, 0] + 0.5, positions[:, 1] + 0.5, positions[:, 2] + 0.5, c="orange", s=60, label="Drones")
        self.ax.legend(loc="upper right")
        return self.drone_scatter,

    def update(self, frame_idx):
        # step all agents once per frame
        for agent in self.agents:
            agent.step()

        # update scanned voxels (subsample to keep performance bounded)
        if self.render_scanned:
            xs, ys, zs = np.where(self.env.scanned == 1)
            if len(xs) > 0:
                n = len(xs)
                max_points = self.max_scanned_points
                stride = max(1, n // max_points)
                xs_s, ys_s, zs_s = xs[::stride] + 0.5, ys[::stride] + 0.5, zs[::stride] + 0.5
            else:
                xs_s = ys_s = zs_s = []
            # update in-place for 3D scatter
            if self.scanned_scatter is None:
                self.scanned_scatter = self.ax.scatter(xs_s, ys_s, zs_s, c="lightblue", alpha=0.2, s=4)
            else:
                self.scanned_scatter._offsets3d = (xs_s, ys_s, zs_s)

        # update targets color (positions remain constant)
        if self.render_targets and self.target_scatter is not None and len(self.env.targets) > 0:
            discovered_mask = np.array([t.found for t in self.env.targets.values()])
            colors = np.where(discovered_mask, "green", "red").tolist()
            self.target_scatter.set_color(colors)

        # update drones positions in-place
        positions = np.array([self.env.get_drone_position(a.drone_id) for a in self.agents])
        if self.drone_scatter is None:
            self.drone_scatter = self.ax.scatter(positions[:, 0] + 0.5, positions[:, 1] + 0.5, positions[:, 2] + 0.5, c="orange", s=60)
        else:
            self.drone_scatter._offsets3d = (
                positions[:, 0] + 0.5,
                positions[:, 1] + 0.5,
                positions[:, 2] + 0.5,
            )

        # annotate telemetry
        telemetry = self.swarm.telemetry()
        coverage_pct = telemetry["coverage"] * 100.0
        remaining = len(telemetry["remaining_targets"])
        self.ax.set_title(f"Coverage: {coverage_pct:.1f}% | Remaining targets: {remaining}")

        # occasional GC to curb fragmentation in long runs
        if (frame_idx % 50) == 0:
            gc.collect()

        return self.scanned_scatter, self.target_scatter, self.drone_scatter


def main():
    cfg, env, swarm, agents = create_swarm_from_config()
    # set up run logger and register each agent
    run_logger = RunLogger()
    for agent in agents:
        run_logger.register_agent(agent.drone_id)

    viz = Visualization(
        env,
        swarm,
        agents,
        max_scanned_points=cfg.visualization.max_scanned_points,
        render_scanned=cfg.visualization.render_scanned,
        render_targets=cfg.visualization.render_targets,
    )
    anim = FuncAnimation(
        viz.fig,
        viz.update,
        init_func=viz.init_draw,
        frames=cfg.visualization.frames,
        interval=cfg.visualization.interval_ms,
        blit=False,
        cache_frame_data=False,
        save_count=50,
    )
    try:
        plt.show()
    finally:
        # flush each agent's memory to its log file
        for agent in agents:
            max_lines = max(100, agent.memory_limit * 2)
            run_logger.flush_agent_memory(agent.drone_id, agent.memory, max_lines=max_lines)


if __name__ == "__main__":
    main()