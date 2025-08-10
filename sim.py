import time

def run_simulation(drones, controller, steps=10):
    """Run simulation for a number of steps."""
    environment = {"target": [50, 50, 0]}  # Mock target (missing person)
    for step in range(steps):
        print(f"Step {step + 1}")
        # Update each drone
        for drone in drones:
            drone.collect_sensor_data(environment)
            status = drone.get_status()
            print(f"{drone.id}: {status}")
        # LLM coordinates based on status
        instruction = "Check all drones' sensor data. Move drones to [50, 50, 0] if a human is detected."
        result = controller.coordinate(instruction)
        print(f"LLM Decision: {result}")
        time.sleep(1)  # Simulate real-time
    return [d.get_status() for d in drones]


