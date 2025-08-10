# Agentic AI Drone Swarm Simulation (LLM-Driven)

A prototype 3D search-and-rescue swarm where each drone is controlled by an LLM agent that uses tools to move, scan, and communicate.

## Features
- LLM-driven agent loop (tool-calling)
- 3D grid environment (numpy)
- Real-time 3D viz (matplotlib animation)
- Swarm manager and simple message bus
- Mock mode (no API) or OpenAI `gpt-4o`

## Install
```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run (Mock LLM)
```
export LLM_PROVIDER=mock
python main.py
```

## Run (OpenAI)
```
export LLM_PROVIDER=openai
export OPENAI_API_KEY=your_key
python main.py
```

## Configuration
- File: `config.json` (override path with `SWARM_CONFIG`)
- Example keys:
  - `swarm.num_drones`: number of agents
  - `environment.size_x|size_y|size_z|num_targets|rng_seed`
  - `visualization.frames|interval_ms|max_scanned_points`
  - `agent.memory_limit`
  - `llm.provider|model|temperature`

```json
{
  "swarm": {"num_drones": 8},
  "environment": {"size_x": 30, "size_y": 30, "size_z": 6, "num_targets": 10, "rng_seed": 7},
  "visualization": {"frames": 600, "interval_ms": 300, "max_scanned_points": 4000},
  "agent": {"memory_limit": 120},
  "llm": {"provider": "mock", "model": "gpt-4o-mini", "temperature": 0.2}
}
```

To use a different config file:
```
export SWARM_CONFIG=/path/to/your_config.json
python main.py
```

## Files
- `agents/drone_agent.py`: LLM agent logic
- `core/environment.py`: 3D world + scan
- `core/swarm.py`: swarm orchestration
- `tools/movement.py`: move/scan tool wrappers
- `main.py`: viz + stepping loop

## Notes
- Assumes a 20x20x5 grid by default
- Simulation pauses between agent decisions
- Easily extend with memory/planning modules 