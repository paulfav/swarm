import numpy as np

class Drone:
    def __init__(self, id, position, battery=100, max_speed=5):
        self.id = id  # Unique identifier (e.g., "Drone_1")
        self.position = np.array(position, dtype=float)  # [x, y, z] coordinates
        self.battery = battery  # Percentage, 0-100
        self.max_speed = max_speed  # Meters per second
        self.sensor_data = {}  # Store simulated sensor data
        self.path = [self.position.copy()]  # Store position history for plotting

    def move_to(self, target):
        """Move drone to target [x, y, z], respecting max speed."""
        direction = np.array(target) - self.position
        distance = np.linalg.norm(direction)
        if distance > self.max_speed:
            direction = direction / distance * self.max_speed
        self.position += direction
        self.battery -= distance * 0.1  # Simple battery drain
        self.path.append(self.position.copy())  # Track path
        return self.position

    def collect_sensor_data(self, environment):
        """Simulate thermal sensor data based on distance to target."""
        target = np.array(environment.get("target", [50, 50, 0]))
        distance = np.linalg.norm(self.position - target)
        self.sensor_data["thermal"] = 37.2 if distance < 10 else 20.0  # Human-like reading if close
        return self.sensor_data

    def get_status(self):
        """Return drone status for control or display."""
        return {
            "id": self.id,
            "position": self.position.tolist(),
            "battery": self.battery,
            "sensor_data": self.sensor_data
        }