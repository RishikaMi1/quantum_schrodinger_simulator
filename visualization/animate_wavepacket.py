import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

hbar = 1.0
m = 1.0

# Spatial grid
N = 400
L = 100
dx = L / N
x = np.linspace(-L/2, L/2, N)

# Time step
dt = 0.01

x0 = -20        # initial position
k0 = 2          # momentum
sigma = 2       # width

psi = np.exp(-(x - x0)**2 / (2 * sigma**2)) * np.exp(1j * k0 * x)

# Normalize
psi = psi / np.sqrt(np.sum(np.abs(psi)**2))

# Potential (free particle)
V = np.zeros(N)

fig, ax = plt.subplots()

line_prob, = ax.plot(x, np.abs(psi)**2, label="Probability Density |ψ|²")
line_real, = ax.plot(x, np.real(psi), linestyle="--", label="Re(ψ)")
line_imag, = ax.plot(x, np.imag(psi), linestyle=":", label="Im(ψ)")

ax.set_xlim(-50, 50)
ax.set_ylim(-1, 1)
ax.set_xlabel("Position")
ax.set_ylabel("Wave Function")
ax.set_title("Quantum Wave Packet Evolution")
ax.legend()

def evolve(psi):

    laplacian = (
        np.roll(psi, -1)
        - 2 * psi
        + np.roll(psi, 1)
    ) / dx**2

    psi = psi + dt * (
        1j * hbar / (2 * m) * laplacian
        - 1j * V * psi / hbar
    )

    return psi

def update(frame):
    global psi

    psi = evolve(psi)

    prob = np.abs(psi)**2

    line_prob.set_ydata(prob)
    line_real.set_ydata(np.real(psi))
    line_imag.set_ydata(np.imag(psi))

    return line_prob, line_real, line_imag
ani = FuncAnimation(
    fig,
    update,
    frames=600,
    interval=30,
    blit=True
)

plt.show()

