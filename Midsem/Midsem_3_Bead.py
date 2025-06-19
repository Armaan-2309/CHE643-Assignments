import numpy as np
import matplotlib.pyplot as plt

# Parameters
v = 250
delta_t_star = 0.001
total_t_star = 10000
steps = int(total_t_star / delta_t_star)
output_interval = 100

# Initial positions
r1_star = np.array([0.0, 0.0, 0.0])
r2_star = np.array([np.sqrt(v), 0.0, 0.0])
r3_star = np.array([2 * np.sqrt(v), 0.0, 0.0])

R_end_star = []
times = []

np.random.seed(42)

# Simulation loop
for step in range(steps + 1):
    t_star = step * delta_t_star

    # SPRING 1:
    R12 = r2_star - r1_star
    R12_mag = np.linalg.norm(R12)
    r_hat_12 = R12_mag / v
    if abs(1 - r_hat_12**2) < 1e-6:
        r_hat_12 = 0.999
    F12 = ((3 - r_hat_12**2) / (v * (1 - r_hat_12**2))) * R12

    # SPRING 2:
    R23 = r3_star - r2_star
    R23_mag = np.linalg.norm(R23)
    r_hat_23 = R23_mag / v
    if abs(1 - r_hat_23**2) < 1e-6:
        r_hat_23 = 0.999
    F23 = ((3 - r_hat_23**2) / (v * (1 - r_hat_23**2))) * R23

    # Brownian motion for each bead
    B1 = np.sqrt(6 / delta_t_star) * np.random.uniform(-1, 1, 3)
    B2 = np.sqrt(6 / delta_t_star) * np.random.uniform(-1, 1, 3)
    B3 = np.sqrt(6 / delta_t_star) * np.random.uniform(-1, 1, 3)

    # Update velocities
    dr1 = B1 + F12
    dr2 = B2 - F12 + F23
    dr3 = B3 - F23

    # Euler integration
    r1_star += dr1 * delta_t_star
    r2_star += dr2 * delta_t_star
    r3_star += dr3 * delta_t_star

    if step % output_interval == 0:
        R_end_star.append(np.linalg.norm(r3_star - r1_star))
        times.append(t_star)

# RMS calculation
R_end_star = np.array(R_end_star)
rms_R_end = np.sqrt(np.mean(R_end_star**2))

with open('R_end_star_3bead_vs_time.txt', 'w') as f:
    f.write('t_star\tR_end_star\n')
    for t, R in zip(times, R_end_star):
        f.write(f'{t:.3f}\t{R:.6f}\n')

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(times, R_end_star, 'b-', linewidth=1, label='R_end*')
plt.axhline(y=rms_R_end, color='k', linewidth=2, linestyle='--', label=f'RMS R_end* ≈ {rms_R_end:.3f}')
plt.xlabel('t*')
plt.ylabel('R_end* = |r3 - r1|')
plt.title('3-Bead Brownian Dynamics (Dimensionless)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('R_end_star_3bead_plot.png')
plt.show()

print(f'RMS value of R_end*: {rms_R_end:.3f}')
