from langchain.agents import initialize_agent, Tool
from langchain.llms import OpenAI
class SwarmController:
    def __init__(self, drones, environment):
        self.drones = drones  # List of Drone objects
        self.environment = environment  # Dict with target location, terrain, etc.
        self.llm = OpenAI(api_key="YOUR_LLM_KEY")
        self.agent = self._initialize_agent()

    def _get_drone_status(self, drone_id: str) -> str:
        """Tool: Get status of a specific drone."""
        drone = next((d for d in self.drones if d.id == drone_id), None)
        return str(drone.get_status()) if drone else "Drone not found"

    def _move_drone(self, input_str: str) -> str:
        """Tool: Move a drone to target coordinates."""
        drone_id, x, y, z = input_str.split(",")
        drone = next((d for d in self.drones if d.id == drone_id), None)
        if drone:
            new_pos = drone.move_to([float(x), float(y), float(z)])
            return f"Drone {drone_id} moved to {new_pos.tolist()}"
        return "Drone not found"

    def _initialize_agent(self):
        tools = [
            Tool(name="GetDroneStatus", func=self._get_drone_status, description="Get status of a drone by ID"),
            Tool(name="MoveDrone", func=self._move_drone, description="Move drone to x,y,z: input as 'id,x,y,z'")
        ]
        return initialize_agent(tools, self.llm, agent_type="zero-shot-react-description")

    def coordinate(self, instruction: str):
        """Run LLM to coordinate swarm based on instruction."""
        return self.agent.run(instruction)
    

def control_swarm(drones, environment):
    """Rule-based controller: Move drones toward target if detected, else search grid."""
    search_grid = [[20, 20, 0], [20, 80, 0], [80, 20, 0], [80, 80, 0]]  # Predefined search points
    for drone in drones:
        status = drone.get_status()
        sensor_data = drone.collect_sensor_data(environment)
        # Rule 1: If human detected (thermal ~37.2°C), move to target
        if abs(sensor_data["thermal"] - 37.2) < 1:
            target = environment["target"]
            print(f"{drone.id} detected human, moving to {target}")
            drone.move_to(target)
        # Rule 2: If battery low (<20%), return to base [0, 0, 0]
        elif status["battery"] < 20:
            print(f"{drone.id} low battery, returning to base")
            drone.move_to([0, 0, 0])
        # Rule 3: Search predefined grid points
        else:
            # Cycle through grid points based on drone ID and time
            grid_idx = (int(drone.id.split('_')[1]) + len(drone.path)) % len(search_grid)
            target = search_grid[grid_idx]
            print(f"{drone.id} searching grid point {target}")
            drone.move_to(target)
    return [drone.get_status() for drone in drones]