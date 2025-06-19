import numpy as np
import matplotlib.pyplot as plt

# Parameters
v = 167
delta_t_star = 0.001
total_t_star = 10000
steps = int(total_t_star / delta_t_star)
output_interval = 100

# Initial positions: aligned along x
r1 = np.array([0.0, 0.0, 0.0])
r2 = np.array([np.sqrt(v), 0.0, 0.0])
r3 = np.array([2*np.sqrt(v), 0.0, 0.0])
r4 = np.array([3*np.sqrt(v), 0.0, 0.0])

R_end_star = []
times = []

np.random.seed(42)

def spring_force(ri, rj):
    R = rj - ri
    R_mag = np.linalg.norm(R)
    r_hat = R_mag / v
    if abs(1 - r_hat**2) < 1e-6:
        r_hat = 0.999 
    return ((3 - r_hat**2) / (v * (1 - r_hat**2))) * R

# Simulation loop
for step in range(steps + 1):
    t_star = step * delta_t_star

    # Brownian motion
    B1 = np.sqrt(6 / delta_t_star) * np.random.uniform(-1, 1, 3)
    B2 = np.sqrt(6 / delta_t_star) * np.random.uniform(-1, 1, 3)
    B3 = np.sqrt(6 / delta_t_star) * np.random.uniform(-1, 1, 3)
    B4 = np.sqrt(6 / delta_t_star) * np.random.uniform(-1, 1, 3)

    # Spring forces
    F12 = spring_force(r1, r2)
    F23 = spring_force(r2, r3)
    F34 = spring_force(r3, r4)

    # Update velocities
    dr1 = B1 + F12
    dr2 = B2 - F12 + F23
    dr3 = B3 - F23 + F34
    dr4 = B4 - F34

    # Euler update
    r1 += dr1 * delta_t_star
    r2 += dr2 * delta_t_star
    r3 += dr3 * delta_t_star
    r4 += dr4 * delta_t_star

    # Save end-to-end distance
    if step % output_interval == 0:
        R_end_star.append(np.linalg.norm(r4 - r1))
        times.append(t_star)

# RMS calculation
R_end_star = np.array(R_end_star)
rms = np.sqrt(np.mean(R_end_star**2))

with open("R_end_star_4bead.txt", "w") as f:
    f.write("t_star\tR_end_star\n")
    for t, R in zip(times, R_end_star):
        f.write(f"{t:.3f}\t{R:.6f}\n")

# Plot
plt.figure(figsize=(10, 6))
plt.plot(times, R_end_star, 'b-', linewidth=1, label="R_end*")
plt.axhline(y=rms, color='k', linestyle='--', linewidth=2, label=f"RMS R_end* ≈ {rms:.3f}")
plt.xlabel("t*")
plt.ylabel("R_end* = |r4 - r1|")
plt.title("4-Bead Brownian Dynamics")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("R_end_star_4bead_plot.png")
plt.show()

print(f'RMS R_end* = {rms:.3f}')
