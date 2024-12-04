import numpy as np
import scipy
from scipy.linalg import eigvals
import matplotlib.pyplot as plt
import seaborn as sns

# LaTeX Formatting

plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.preamble'] = r'\usepackage{bm, amssymb, amsmath}'
# If necessary, specify the LaTeX installation path
# plt.rcParams['text.latex.unicode'] = True
plt.rcParams['font.weight'] = 'bold'

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 19

# Standard constants and matrices.

kappa = 0.5  # penalising factor (set in advance)
n = 5  # no of qubits
tmax = [1, 5, 10, 30, 100]  # total runtime of the algorithm
q = 200  # number of timesteps
i = complex(0, 1)  # complex i

J = np.array([[0, 1, 1, 0, 0],
              [0, 0, 1, 0, 1],
              [0, 0, 0, 1, 1],
              [0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0]])

h = np.array([[-2 + kappa],
              [-3 + kappa],
              [-4 + kappa],
              [-1 + kappa],
              [-2 + kappa]])

sigma_z = np.array([[1, 0],
                    [0, -1]])

sigma_x = np.array([[0, 1],
                    [1, 0]])

I_2 = np.eye(2)


def sigma_z_j(n, j):
    " Tensor product for sigma^z_j, where n is the total number of qubits in the system and the Pauli-Z gate is applied to the jth qubit "
    calc = np.eye(1)  # Intialise calculation, apply identity to the first (j-1) qubits

    for i in range(1, j):
        calc = np.kron(calc, I_2)

    # Apply σ^z at the j-th position
    calc = np.kron(calc, sigma_z)

    # Tensor product of (n-j) identity matrices from (j+1) to nth qubit
    for i in range(j + 1, n + 1):
        calc = np.kron(calc, I_2)

    calc[np.isclose(calc, 0)] = 0  # Changes all instances of -0. to 0.
    return calc


def sigma_x_j(n, j):
    " Tensor product for sigma^x_j, where n is the total number of qubits in the system and the Pauli-X gate is applied to the jth qubit "
    calc = np.eye(1)  # Intialise calculation, apply identity to the first (j-1) qubits

    for i in range(1, j):
        calc = np.kron(calc, I_2)

    # Apply σ^x at the j-th position
    calc = np.kron(calc, sigma_x)

    # Tensor product of (n-j) identity matrices from (j+1) to nth qubit
    for i in range(j + 1, n + 1):
        calc = np.kron(calc, I_2)

    calc[np.isclose(calc, 0)] = 0  # Changes all instances of -0. to 0.
    return calc


def H_Ising(J, h):
    " Produces time-independent Ising matrix from graph J and penalising matrix h "

    H_problem = np.zeros((2 ** n, 2 ** n))  # Size of final Hamiltonian

    for k in range(n):
        for j in range(k + 1, n):
            if J[k, j] != 0:  # Only compute if J_kj is non-zero
                sigma_kz = sigma_z_j(n, k + 1)
                sigma_jz = sigma_z_j(n, j + 1)
                H_problem += J[k, j] * np.matmul(sigma_kz, sigma_jz)

    for j in range(n):
        if h[j, 0] != 0:  # Only compute if h_j is non-zero
            sigma_jz = sigma_z_j(n, j + 1)
            H_problem += h[j, 0] * sigma_jz

    return H_problem


#-------------------------------------------------------------------------------------------------------------------------------------------


# Initialise t=0 ground state
def initial_state(n):
    return np.ones(2 ** n) / np.sqrt(2 ** n)


psi_0 = initial_state(n)  # psi at t=0

# Encode maximum independent set - will write proper function for this

max_independent_set = np.zeros(2 ** n)
max_independent_set[19] = 1


# Create born rule function - this is for calculating success probability and fidelity
def born_rule(state, target_state):
    inner_product = np.vdot(target_state, state)  # Inner product <target_state|state>
    probability = np.abs(inner_product) ** 2  # Square of the absolute value
    return probability


# Calculate energy eigenvectors and eigenvalues after each evolution step

def energy(J, h, ground_state, target_state, tmax_value, q):
    step = tmax_value / q
    psi = psi_0

    timesteps = np.arange(0, tmax_value, step)

    ground_energies = np.zeros(len(timesteps))
    first_excited_energies = np.zeros(len(timesteps))
    energy_gap = np.zeros(len(timesteps))

    for i, t in enumerate(timesteps):
        # the following functions (A and B) are annealing schedules
        A = 1 - t / tmax_value
        B = t / tmax_value

        H_evolve = np.zeros((2 ** n, 2 ** n))

        H_ising_term = H_Ising(J, h)
        H_interaction = B * H_ising_term

        for j in range(1, n + 1):
            H_evolve -= A * sigma_x_j(n, j)

        H_t = H_evolve + H_interaction

        eigenvalues = np.sort(scipy.linalg.eigh(H_t, eigvals_only=True))

        ground_energies[i] = eigenvalues[0]
        first_excited_energies[i] = eigenvalues[1]

        # energy gap between ground state and first excited states
        energy_gap[i] = eigenvalues[1] - eigenvalues[0]

    return energy_gap, ground_energies, first_excited_energies

energy_gaps = energy(J, h, psi_0, max_independent_set, 100, q)[0]
print("Minimum =", np.min(energy_gaps)) # Minimum energy gap, which is independent of tmax_value

plt.figure(figsize=(7,6))

timesteps = np.linspace(0, 100, q)

#plt.plot(timesteps, energy(J, h, psi_0, max_independent_set, 1, q)[0], label='Difference')
plt.plot(timesteps, energy(J, h, psi_0, max_independent_set, 100, q)[0], linestyle='--', label='Difference')
plt.plot(timesteps, energy(J, h, psi_0, max_independent_set, 100, q)[1], label='Ground State')
plt.plot(timesteps, energy(J, h, psi_0, max_independent_set, 100, q)[2], label='First Excited State')
plt.title("Minimum spectral gap $g_{{min}}$ scaled with time")
plt.xlabel("Timestep")
plt.ylabel("Energy")
plt.legend()
plt.show()
