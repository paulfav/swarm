from __future__ import annotations

import json
import os
from typing import Optional
from pydantic import BaseModel, Field


class LLMConfig(BaseModel):
    provider: str = Field(default="mock")  # mock|openai
    model: str = Field(default="gpt-4o-mini")
    temperature: float = Field(default=0.2)


class EnvironmentConfig(BaseModel):
    size_x: int = 20
    size_y: int = 20
    size_z: int = 5
    num_targets: int = 6
    rng_seed: int = 7


class SwarmConfig(BaseModel):
    num_drones: int = 5


class VisualizationConfig(BaseModel):
    frames: int = 500
    interval_ms: int = 400
    max_scanned_points: int = 3000
    render_scanned: bool = True
    render_targets: bool = True


class AgentConfig(BaseModel):
    memory_limit: int = 100


class AppConfig(BaseModel):
    llm: LLMConfig = LLMConfig()
    environment: EnvironmentConfig = EnvironmentConfig()
    swarm: SwarmConfig = SwarmConfig()
    visualization: VisualizationConfig = VisualizationConfig()
    agent: AgentConfig = AgentConfig()


def load_config(config_path: Optional[str] = None) -> AppConfig:
    path = config_path or os.getenv("SWARM_CONFIG", os.path.join(os.getcwd(), "config.json"))
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return AppConfig(**data)
    return AppConfig() 