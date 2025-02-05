import numpy as np
import matplotlib.pyplot as plt
import random
import scipy.sparse as sps

# Develop the Ising Problem Hamiltonian based on bipartite diamond system

import numpy as np
import random
import matplotlib.pyplot as plt
import scipy.sparse as sps

# Define the coupling list
kvp = [[1, 5, 1.0],
       [1, 6, 1.0],
       [1, 7, 1.0],
       [1, 8, 1.0],
       [2, 5, 1.0],
       [2, 6, 1.0],
       [2, 7, 1.0],
       [2, 8, 1.0],
       [3, 5, 1.0],
       [3, 6, 1.0],
       [3, 7, 1.0],
       [3, 8, 1.0],
       [4, 5, 1.0],
       [4, 6, 1.0],
       [4, 7, 1.0],
       [4, 8, 1.0],
       [9, 13, 1.0],
       [9, 14, 1.0],
       [9, 15, 1.0],
       [9, 16, 1.0],
       [10, 13, 1.0],
       [10, 14, 1.0],
       [10, 15, 1.0],
       [10, 16, 1.0],
       [11, 13, 1.0],
       [11, 14, 1.0],
       [11, 15, 1.0],
       [11, 16, 1.0],
       [12, 13, 1.0],
       [12, 14, 1.0],
       [12, 15, 1.0],
       [12, 16, 1.0],
       [5, 13, 1.0],
       [6, 14, 1.0],
       [7, 15, 1.0],
       [8, 16, 1.0],
       [1, 1, 1.0],
       [2, 2, 1.0],
       [3, 3, 1.0],
       [4, 4, 1.0],
       [5, 5, 1.0],
       [6, 6, 1.0],
       [7, 7, 1.0],
       [8, 8, 1.0],
       [9, 9, -1.0],
       [10, 10, -1.0],
       [11, 11, -1.0],
       [12, 12, -1.0],
       [13, 13, -1.0],
       [14, 14, -1.0],
       [15, 15, -1.0],
       [16, 16, -1.0]]

# Define the number of spins
nspins = 16

# Set Ising matrix as a sparse DOK matrix
isingJ = sps.dok_matrix((nspins, nspins))

# Populate isingJ
for i, j, val in kvp:
    isingJ[i-1, j-1] = val

# Convert to a dense matrix (easier to visualise) and convert to symmetric matrix
dense_isingJ = isingJ.toarray()
dense_isingJ = dense_isingJ + dense_isingJ.T

print("Dense Ising matrix:")
print(dense_isingJ)

def classical_ising_energy(spin_state, J):
    """ Compute the classical Ising energy for a given spin configuration NOTE: modify this for h matrix too """
    spin_state = np.array(spin_state)  # Ensure it's an array
    energy = 0.0

    # Iterate over all pairs i, j (only count unique pairs)
    for i in range(len(spin_state)):
        for j in range(len(spin_state)):
            if i == j:  # Local field term (diagonal of J)
                energy -= J[i, j] * spin_state[i]
            elif i < j:  # Interaction term (only count once)
                energy -= J[i, j] * spin_state[i] * spin_state[j]
    return energy

def simulated_annealing(J, nspins, initial_temp=3, final_temp=0, annealing_steps=100, mcsteps=1):
    """Perform simulated annealing on the Ising system."""

    spins = np.random.choice([-1, 1], size=nspins) # Local quantum annealing: instead of generating a randomised new Ising system, we generate a random Spin state
    print("Initialised state:", spins)

    temperature_range = np.linspace(initial_temp, final_temp, annealing_steps) # Set annealing schedule to be linear

    energy_history = [] # Determining energy of state at each iteration
    current_energy = classical_ising_energy(spins, J)
    energy_history.append(current_energy)

    for T in temperature_range:
        for _ in range(mcsteps):
            # Pick a random spin to flip
            i = random.randint(0, nspins - 1)
            new_spins = spins.copy()
            new_spins[i] *= -1  # Flip spin
            print(f"{i}: {new_spins}")

            # Calculate energy difference between our "current energy" and energy of the spin state we just created
            new_energy = classical_ising_energy(new_spins, J)
            delta_E = new_energy - current_energy

            # Apply Metropolis condition; if energy diff less than 0 accept, else accept with probability exp(-delta_E / T)
            if delta_E < 0 or np.random.uniform() < np.exp(-delta_E / T):
                spins = new_spins        # Accept new configuration
                current_energy = new_energy

            # Store the energy at each MC step
            energy_history.append(current_energy)
    print("Ground state:", spins)
    return energy_history

# Run Simulated Annealing
energy_history = simulated_annealing(dense_isingJ, nspins)

# Plot Energy vs. Monte Carlo Steps
plt.figure(figsize=(8, 5))
plt.plot(energy_history, label="Simulated Annealing Energy")
plt.xlabel("Monte Carlo Steps")
plt.ylabel("Energy")
plt.title("Energy vs. Monte Carlo Steps")
plt.legend()
plt.show()

# ----- Compute and Plot Residual Energy -----
# Here we define the residual energy at each step as the difference between the energy at that step and the minimum energy reached.
min_energy = min(energy_history)
residual_energy = np.array(energy_history) - min_energy

plt.figure(figsize=(8, 5))
plt.plot(residual_energy, label="Residual Energy")
plt.xlabel("Monte Carlo Steps")
plt.ylabel("Residual Energy (E - E_min)")
plt.title("Residual Energy vs. Monte Carlo Steps")
plt.legend()
plt.show()

def smooth_energy(energy_history, window_size=10):
    smoothed = np.convolve(energy_history, np.ones(window_size)/window_size, mode='valid')
    return smoothed

smoothed_residual = smooth_energy(residual_energy, window_size=10)
plt.figure(figsize=(8, 5))
plt.plot(smoothed_residual, label="Smoothed Residual Energy")
plt.xlabel("Monte Carlo Steps (smoothed)")
plt.ylabel("Residual Energy")
plt.title("Smoothed Residual Energy vs. Monte Carlo Steps")
plt.legend()
plt.show()
