import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_swarm(drones, environment):
    """Plot drone positions and target in 3D."""
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    # Plot drones
    for drone in drones:
        pos = drone.position
        ax.scatter(pos[0], pos[1], pos[2], label=drone.id)
    # Plot target (missing person)
    target = environment.get("target", [50, 50, 0])
    ax.scatter(target[0], target[1], target[2], c="red", marker="x", s=100, label="Target")
    # Set labels
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()
    plt.show()

