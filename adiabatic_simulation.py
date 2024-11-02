import numpy as np
import scipy
import matplotlib.pyplot as plt

# Standard constants and matrices. J  is the graph defined by a 3-qubit problem given in Appendix 3

kappa = 0.5 # penalising factor (set in advance)
n = 3 # no of qubits
tmax = [1, 2, 5, 10, 100] # total runtime of the algorithm
q = 200 # number of timesteps
i = complex(0,1)


J = np.array([[0, 1, 0],
              [0, 0, 1],
              [0, 0, 0]])

h = np.array([[-1 + kappa],
              [-2 + kappa],
              [-1 + kappa]])

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

    H_problem = np.zeros((2 ** n, 2 ** n))  # Size of final Hamiltonian

    for k in range(n):
        for j in range(k+1, n):
            if J[k, j] != 0:  # Only compute if J_kj is non-zero
                sigma_kz = sigma_z_j(n, k+1)
                sigma_jz = sigma_z_j(n, j+1)
                H_problem += J[k, j] * np.matmul(sigma_kz, sigma_jz)
                #print(J[k, j], sigma_kz, sigma_jz)

    for j in range(n):
        if h[j, 0] != 0:  # Only compute if h_j is non-zero
            sigma_jz = sigma_z_j(n, j+1)
            H_problem += h[j, 0] * sigma_jz

    return H_problem


# Initialise t=0 state:
def initial_state(n):
    return np.ones(2**n) / np.sqrt(2**n)

psi_0 = initial_state(n) # psi at t=0

# Encoding maximum independent set:
max_independent_set = np.zeros(2**n)
max_independent_set[5] = 1

def born_rule(state, target_state):
    inner_product = np.vdot(target_state, state)  # inner product <target_state|state>
    probability = np.abs(inner_product) ** 2
    return probability


def time_evolution_operator(J, h, ground_state, target_state, tmax_value, q):
    step = tmax_value/q
    psi = psi_0
    probabilities = []

    timesteps = np.arange(0, tmax_value, step)

    for t in timesteps:
        A = 1 - t/tmax_value
        B = t/tmax_value

        H_evolve = np.zeros((2 ** n, 2 ** n))

        H_ising_term = H_Ising(J, h)
        H_interaction = B * H_ising_term

        for j in range(1, n+1):
            H_evolve -= A * sigma_x_j(n, j)

        H_t = H_evolve + H_interaction

        psi = np.matmul(scipy.linalg.expm(-1 * i * step * H_t), psi)
        prob = born_rule(psi, target_state)
        probabilities.append(prob)

    return probabilities


plt.figure(figsize=(8,6))

for tmax_value in tmax:
    success_probability = time_evolution_operator(J, h, psi_0, max_independent_set, tmax_value, q)

    x_values = np.arange(0, tmax_value, tmax_value/q) / tmax_value
    y_values = success_probability

    plt.plot(x_values, y_values, label=f'tmax = {tmax_value}')

plt.xlabel('t/tmax')
plt.ylabel('success probability')
plt.legend()
plt.show()








