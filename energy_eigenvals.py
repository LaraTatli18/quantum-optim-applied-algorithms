import numpy as np
import scipy
from scipy.linalg import eigvals
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.preamble'] = r'\usepackage{bm, amssymb, amsmath}'
# If necessary, specify the LaTeX installation path
# plt.rcParams['text.latex.unicode'] = True
plt.rcParams['font.weight'] = 'bold'

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 19

# Standard constants and matrices.

kappa = 0.5 # penalising factor (set in advance)
n = 5 # no of qubits
tmax = [1, 2, 5, 10, 100] # total runtime of the algorithm
q = 200 # number of timesteps
i = complex(0,1) # complex i


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

def sigma_z_j(n,j):
    " Tensor product for sigma^z_j, where n is the total number of qubits in the system and the Pauli-Z gate is applied to the jth qubit "
    calc = np.eye(1) # Intialise calculation, apply identity to the first (j-1) qubits

    for i in range(1, j):
        calc = np.kron(calc, I_2)

    # Apply σ^z at the j-th position
    calc = np.kron(calc, sigma_z)

    # Tensor product of (n-j) identity matrices from (j+1) to nth qubit
    for i in range(j + 1, n + 1):
        calc = np.kron(calc, I_2)

    calc[np.isclose(calc, 0)] = 0 # Changes all instances of -0. to 0.
    return calc

def sigma_x_j(n,j):
    " Tensor product for sigma^x_j, where n is the total number of qubits in the system and the Pauli-X gate is applied to the jth qubit "
    calc = np.eye(1) # Intialise calculation, apply identity to the first (j-1) qubits

    for i in range(1, j):
        calc = np.kron(calc, I_2)

    # Apply σ^x at the j-th position
    calc = np.kron(calc, sigma_x)

    # Tensor product of (n-j) identity matrices from (j+1) to nth qubit
    for i in range(j + 1, n + 1):
        calc = np.kron(calc, I_2)

    calc[np.isclose(calc, 0)] = 0 # Changes all instances of -0. to 0.
    return calc

def H_Ising(J, h):
    " Produces time-independent Ising matrix from graph J and penalising matrix h "

    H_problem = np.zeros((2 ** n, 2 ** n))  # Size of final Hamiltonian

    for k in range(n):
        for j in range(k+1, n):
            if J[k, j] != 0:  # Only compute if J_kj is non-zero
                sigma_kz = sigma_z_j(n, k+1)
                sigma_jz = sigma_z_j(n, j+1)
                H_problem += J[k, j] * np.matmul(sigma_kz, sigma_jz)

    for j in range(n):
        if h[j, 0] != 0:  # Only compute if h_j is non-zero
            sigma_jz = sigma_z_j(n, j+1)
            H_problem += h[j, 0] * sigma_jz

    return H_problem


# Initialise t=0 state
def initial_state(n):
    return np.ones(2**n) / np.sqrt(2**n)

psi_0 = initial_state(n) # psi at t=0

# Encode maximum independent set - will write proper function for this

max_independent_set = np.zeros(2**n)
max_independent_set[19] = 1


def born_rule(state, target_state):
    inner_product = np.vdot(target_state, state)  # Inner product <target_state|state>
    probability = np.abs(inner_product) ** 2  # Square of the absolute value
    return probability

initial_overlap = born_rule(psi_0, max_independent_set)
print("Initial overlap probability:", initial_overlap)

def calculate_fidelity(J, h, ground_state, target_state, tmax_value, q):
    step = tmax_value/q
    psi = psi_0
    fidelities = []

    timesteps = np.arange(0, tmax_value, step)

    for t in timesteps:
        # the following functions (A and B) are annealing schedules
        A = 1 - t/tmax_value
        B = t/tmax_value

        H_evolve = np.zeros((2 ** n, 2 ** n))

        H_ising_term = H_Ising(J, h)
        H_interaction = B * H_ising_term

        for j in range(1, n+1):
            H_evolve -= A * sigma_x_j(n, j)

        H_t = H_evolve + H_interaction

        e_vals, e_vectors = scipy.linalg.eigh(H_t)
        groundState = e_vectors[:,0]

        psi = np.matmul(scipy.linalg.expm(-1 * i * step * H_t), psi) # evolution of the state
        fidelity = born_rule(psi, groundState) # calculate fidelity

        fidelities.append(fidelity)

    return fidelities

def ground_state_energy(J, h, ground_state, target_state, tmax_value, q):
    step = tmax_value / q
    psi = psi_0
    energy_gaps = []

    timesteps = np.arange(0, tmax_value, step)

    for t in timesteps:
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
        print("eigenvalues =", eigenvalues)
        ground_state_energy = eigenvalues[0]
        print("ground state energy =", ground_state_energy)
        first_excited_energy = eigenvalues[1]
        print("first excited energy =", first_excited_energy)

        # energy gap between ground state and first excited states
        energy_gap = first_excited_energy - ground_state_energy
        energy_gaps.append(energy_gap)

    return energy_gaps

# Plot ground state energy as a function of time

timesteps = np.linspace(0, 1, q)
plt.plot(timesteps, ground_state_energy(J, h, psi_0, max_independent_set, 100, q))
plt.title("Minimum spectral gap $g_{{min}}$ scaled with time")
plt.show()

# Plot fidelity against runtime

plt.figure(figsize=(8,6))
colors = sns.color_palette("hls", len(tmax))

for tmax_value, color in zip(tmax, colors):
    success_probability = calculate_fidelity(J, h, psi_0, max_independent_set, tmax_value, q)

    x_values = np.arange(0, tmax_value, tmax_value/q) / tmax_value
    y_values = success_probability

    plt.plot(x_values, y_values, label=f'$t_{{max}}$ = {tmax_value}', color=color)

plt.xlabel('$t$/$t_{{max}}$')
plt.ylabel('Fidelity')
#plt.title("Success Probability scaled with algorithm runtime for 5-qubit Graph")
plt.legend(title = "Algorithm runtime")
plt.savefig('fidelity_plot.svg', transparent=True)
plt.show()

fig, ax1 = plt.subplots()

color = 'tab:blue'
ax1.set_xlabel('t/tmax')
ax1.set_ylabel('Fidelity')

for tmax_value, color in zip(tmax, colors):
    success_probability = calculate_fidelity(J, h, psi_0, max_independent_set, tmax_value, q)

    x_values = np.arange(0, tmax_value, tmax_value/q) / tmax_value
    y_values = success_probability

    ax1.plot(x_values, y_values, label = f'$t_{{max}}$ = {tmax_value}', color=color)

ax1.tick_params(axis='y', labelcolor='b')

ax2 = ax1.twinx()

ax2.set_ylabel('energy difference')
ax2.plot(timesteps, ground_state_energy(J, h, psi_0, max_independent_set, tmax_value, q), linestyle='--', label=f'$g_{{min}}$ = {tmax_value}')
ax2.tick_params(axis='y', labelcolor='r')

fig.tight_layout()
plt.show()
