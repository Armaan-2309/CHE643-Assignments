import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

# Single Particle Simulation
def simulate_single_particle(t_total=100, dt=0.001):
    N = int(t_total / dt)
    pos = np.zeros((N + 1, 3))

    for i in range(1, N + 1):
        n = np.random.uniform(-1, 1, 3)
        delta_r = 6 * np.sqrt(dt) * n
        pos[i] = pos[i - 1] + delta_r

    return pos

def compute_msd(pos, dt, t_max=10):
    steps = int(t_max / dt)
    msd = []
    for tau in range(1, steps + 1):
        disp = pos[tau:] - pos[:-tau]
        sq_disp = np.sum(disp ** 2, axis=1)
        msd.append(np.mean(sq_disp))
    return np.arange(1, steps + 1) * dt, np.array(msd)

# Simulate single particle
dt = 0.001
pos1 = simulate_single_particle(100, dt)

# Save to file
np.savetxt("trajectory_single_particle.txt", np.column_stack((np.arange(0, 100 + dt, dt), pos1)), header="t* x* y* z*")

# MSD and diffusivity
t_msd, msd_vals = compute_msd(pos1, dt, 10)
slope, intercept, *_ = linregress(t_msd[:1000], msd_vals[:1000])
D_star = slope / 6
print(f"Estimated diffusivity D* = {D_star:.4f}")

# Plot MSD
plt.figure()
plt.plot(t_msd, msd_vals)
plt.xlabel("t*")
plt.ylabel("MSD")
plt.title("MSD vs t* (Single Particle)")
plt.grid(True)
plt.savefig("msd_single_particle.png")

# 100 Particles Simulation
def simulate_many_particles(N=100, t_total=50, dt=0.001, checkpoints=[10, 20, 30, 40, 50]):
    steps = int(t_total / dt)
    record_steps = [int(t / dt) for t in checkpoints]
    pos_all = np.zeros((N, len(record_steps), 3))

    for i in range(N):
        pos = np.zeros((steps + 1, 3))
        for j in range(1, steps + 1):
            n = np.random.uniform(-1, 1, 3)
            delta_r = 6 * np.sqrt(dt) * n
            pos[j] = pos[j - 1] + delta_r
        pos_all[i] = pos[record_steps]

    return pos_all

positions = simulate_many_particles()
times = [10, 20, 30, 40, 50]
avg_dist = [np.mean(np.linalg.norm(positions[:, i, :], axis=1)) for i in range(len(times))]

# Save average distances
np.savetxt("avg_distance_vs_time.txt", np.column_stack((times, avg_dist)), header="t*  average_distance")

# Plot
plt.figure()
plt.plot(times, avg_dist, 'o-')
plt.xlabel("t*")
plt.ylabel("Average Distance from Origin")
plt.title("Average Distance vs t* (100 particles)")
plt.grid(True)
plt.savefig("avg_distance_vs_time.png")
