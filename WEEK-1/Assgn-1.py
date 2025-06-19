import numpy as np
import matplotlib.pyplot as plt

# Constants
R = 0.005               # radius (m)
rho_sphere = 2500       # kg/m³
rho_fluid = 1000        # kg/m³
eta = 0.1               # viscosity (Pa.s)
g = 9.81                # gravity (m/s^2)

# Derived quantities
V = (4/3) * np.pi * R**3              # volume of sphere
m = rho_sphere * V                    # mass of sphere
k = 6 * np.pi * eta * R               # drag coefficient
tau = m / k                           # time constant
u_terminal = (m * g - V * rho_fluid * g) / k

# Time parameters
t_max = 5       
dt = 0.01
t = np.arange(0, t_max + dt, dt)

# Analytical solution
u_analytical = u_terminal * (1 - np.exp(-t / tau))

# Numerical solution using Euler's method
u_numerical = np.zeros_like(t)
for i in range(1, len(t)):
    du = g - (V * rho_fluid * g) / m - (k * u_numerical[i-1]) / m
    u_numerical[i] = u_numerical[i-1] + dt * du

# Plotting
plt.figure(figsize=(8,5))
plt.plot(t, u_analytical, label='Analytical', color='blue')
plt.plot(t, u_numerical, '--', label='Numerical (Euler)', color='red')
plt.xlabel('Time (s)')
plt.ylabel('Velocity (m/s)')
plt.title('Velocity vs Time for a Sphere Falling in Liquid')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
