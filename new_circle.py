import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def euler_method(M, z0, dt, t_max, ring_length):
    """
    Solves z' = Mz using the Euler method with periodic boundary conditions for a ring.

    Parameters:
        M: np.ndarray
            The system matrix.
        z0: np.ndarray
            Initial state vector.
        dt: float
            Time step.
        t_max: float
            Maximum simulation time.
        ring_length: float
            The length of the ring.

    Returns:
        t: np.ndarray
            Array of time points.
        z: np.ndarray
            Array of state vectors at each time point.
    """
    n_steps = int(t_max / dt)
    n_states = len(z0)

    t = np.linspace(0, t_max, n_steps)
    z = np.zeros((n_steps, n_states))
    z[0] = z0

    for i in range(1, n_steps):
        z_next = z[i - 1] + dt * (M @ z[i - 1])
        z[i] = z_next

    return t, z

# Parameters
N = 5  # Number of autonomous cars
b = 1.25  # Proportional gain
c = 1  # Derivative gain
T = 1.0  # Desired time interval
ring_length = 800  # Length of the ring

# Define matrices A and B
A = np.array([[0, 1],
              [-c, -b]])
B = np.array([[0, 0],
              [c, -c * T + b]])

# Construct M matrix
M = np.zeros((2 * N, 2 * N))
for i in range(N):
    M[2 * i:2 * i + 2, 2 * i:2 * i + 2] = A
    M[2 * i:2 * i + 2, 2 * ((i + 1) % N):2 * ((i + 1) % N) + 2] = B  # Connect to the next car (modulo for ring)

initial_dist = 1

# Initial conditions
z0 = np.zeros(2 * N)  # Initial positions and velocities
z0[::2] = np.ones(N)*initial_dist  # Initial positions with 20 meters apart
z0[-2] = ring_length - (N-1)*initial_dist
z0[1::2] = 0  # Initial velocities set to 10 m/s

#print(z0)

# Simulation parameters
dt = 0.01  # Time step
t_max = 30  # Maximum simulation time

# Solve the system
t, z = euler_method(M, z0, dt, t_max, ring_length)

# Extract positions and velocities of each car
d = z[:, ::2]  # Distances
v = z[:, 1::2]  # Velocities

d_cumsum = []
for car in d:
    d_cumsum.append(np.cumsum(car))

# Plot results
plt.figure(figsize=(12, 6))

# Plot distances
for i in range(N):
    plt.plot(t, d[:, i], label=f"Car {i + 1} distance")
plt.title("Distance to car in front")
plt.xlabel("Time (s)")
plt.ylabel("Distance (m)")
plt.legend()
plt.grid()
plt.show()

# Plot velocities
plt.figure(figsize=(12, 6))
for i in range(N):
    plt.plot(t, v[:, i], label=f"Car {i + 1} velocity")
plt.title("Relative velocities of cars over time")
plt.xlabel("Time (s)")
plt.ylabel("Delta v")
plt.legend()
plt.grid()
plt.show()



# Animation of cars on a ring
def animate_cars():
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(-ring_length / 2 - ring_length / 10, ring_length / 2 + ring_length / 10)
    ax.set_ylim(-ring_length / 2 - ring_length / 10, ring_length / 2 + ring_length / 10)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Animation of Cars on a Ring")

    # Draw the ring
    circle = plt.Circle((0, 0), ring_length / 2, color='gray', fill=False, linestyle='--', label="Track")
    ax.add_artist(circle)

    # Convert positions to angles on a ring
    def positions_to_ring(positions):
        angles = (positions / ring_length) * 2 * np.pi
        x = (ring_length / 2) * np.cos(angles)
        y = (ring_length / 2) * np.sin(angles)
        return x, y

    cars, = ax.plot([], [], 'o', markersize=10, label="Cars")

    def init():
        cars.set_data([], [])
        return cars,

    def update(frame):
        x, y = positions_to_ring(d_cumsum[frame])
        cars.set_data(x, y)
        return cars,

    ani = animation.FuncAnimation(fig, update, frames=len(t), init_func=init, blit=True, interval=dt * 50)

    # Save the animation as a GIF
    #from matplotlib.animation import PillowWriter
    #ani.save("cars_on_ring.gif", writer=PillowWriter(fps=30))
    plt.legend()
    plt.show()


animate_cars()


