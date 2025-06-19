import numpy as np
import matplotlib.pyplot as plt

# Parameters
v = 500  # number of Kuhn lengths
delta_t_star = 0.01  # time step
total_t_star = 10000  # total simulation time
steps = int(total_t_star / delta_t_star)
output_interval = 100


r1_star = np.array([0.0, 0.0, 0.0])  # bead 1
r2_star = np.array([np.sqrt(v), 0.0, 0.0])  # bead 2
R_end_star = []
times = []

# Simulation
np.random.seed(42)
for step in range(steps + 1):
    t_star = step * delta_t_star
    
    R_star = r2_star - r1_star
    R_star_mag = np.linalg.norm(R_star)
    r_hat = R_star_mag / v
    
    if abs(1 - r_hat**2) < 1e-6:
        r_hat = 0.999
    
    # Spring force
    spring_term = (3 - r_hat*2) / (v * (1 - r_hat*2)) * R_star
    
    # Brownian motion
    brownian1 = np.sqrt(6 / delta_t_star) * np.random.uniform(-1, 1, 3)
    brownian2 = np.sqrt(6 / delta_t_star) * np.random.uniform(-1, 1, 3)
    
    # Update positions
    dr1_star_dt = brownian1 + spring_term
    dr2_star_dt = brownian2 - spring_term
    
    # Euler integration
    r1_star = r1_star + dr1_star_dt * delta_t_star
    r2_star = r2_star + dr2_star_dt * delta_t_star
    
    if step % output_interval == 0:
        R_end_star.append(R_star_mag)
        times.append(t_star)

# Calculate
R_end_star = np.array(R_end_star)
mean_square = np.mean(R_end_star**2)
rms_R_end = np.sqrt(mean_square)

# Save data to file
with open('R_end_star_vs_time.txt', 'w') as f:
    f.write('t_star\tR_end_star\n')
    for t, R in zip(times, R_end_star):
        f.write(f'{t}\t{R}\n')

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(times, R_end_star, 'b-', linewidth=1, label='R_end*')
plt.axhline(y=rms_R_end, color='k', linewidth=2, label=f'RMS R_end* = {rms_R_end:.3f}')
plt.xlabel('t*')
plt.ylabel('R_end*')
plt.title('End-to-End Distance vs Time')
plt.legend()
plt.grid(True)
plt.savefig('R_end_star_plot.png')
plt.show()

print(f'RMS value of R_end*: {rms_R_end:.3f}')
