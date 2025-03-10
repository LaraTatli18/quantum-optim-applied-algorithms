import numpy as np
import math
import random

def GenerateBoard(N):
    """Generates NxN chessboard with N queens placed at random locations
    1: Queen located
    -1: No queen
    """
    board = -1 * np.ones([N, N], dtype=int)
    # Choose N distinct cells from the N*N cells.
    cells = random.sample(range(N * N), N)
    for cell in cells:
        row = cell // N
        col = cell % N
        board[row, col] = 1
    return board

def PrintReadableBoard(board):
    """Prints the chessboard in a human-friendly format"""
    board_dim = board.shape[0]

    for i in range(board_dim):
        line = ""
        for j in range(board_dim):
            if board[i][j] == 1:
                line += "Q "
            else:
                line += ". "
        print(line)
    print()

def ClassicalEnergy(N, board, A, B, C):
    """ Calculates the potential energy based on an Ising-like model with row, column & diagonal penalties and scaling factors A,B and C"""
    energy = 0.0

    # Row constraints
    for i in range(N):
        row_count = np.sum((1 + board[i, :]) // 2)
        energy += A * ((row_count - 1) ** 2)
    # Column constraints
    for j in range(N):
        col_count = np.sum((1 + board[:, j]) // 2)
        energy += B * ((col_count - 1) ** 2)
    # Main diagonals (NW-SE)
    for d in range(-N + 1, N):
        diag = np.diag(board, k=d)
        diag_count = np.sum((1 + diag) // 2)
        energy += C * (diag_count * (diag_count - 1))
    # Anti-diagonals (NE-SW)
    flipped = np.fliplr(board)
    for d in range(-N + 1, N):
        diag = np.diag(flipped, k=d)
        diag_count = np.sum((1 + diag) // 2)
        energy += C * (diag_count * (diag_count - 1))

    return energy

def SwapMove(board):
    """ Given a board with exactly N queens (cells with +1) and the rest -1,
        perform a swap move: select one cell that has a queen and one empty cell,
        then swap them. Returns the new board and the swapped positions. """
    board_new = board.copy()
    board_dim = board_new.shape[0]

    queen_positions = list(zip(*np.where(board == 1)))
    empty_positions = list(zip(*np.where(board == -1)))
    if not queen_positions or not empty_positions:
        return board_new, None

    pos_queen = random.choice(queen_positions)
    pos_empty = random.choice(empty_positions)

    board_new[pos_queen] = -1
    board_new[pos_empty] = 1

    return board_new, (pos_queen, pos_empty)

def QuantumSwapMove(board_stack, slice_index):
    new_board_stack = board_stack.copy()
    new_board, _ = SwapMove(board_stack[slice_index])
    new_board_stack[slice_index] = new_board

    return new_board_stack

def QuantumEnergy(N, board_stack, A, B, C, J_perp):
    """Calculates the potential energy summed over Trotter slices and the interactions between them.
    board_stack has shape (P, board_dim, board_dim). board_dim = N. Periodic boundaries?
    Coupling here is defined as the sum of the products of corresponding spins. """

    # energy = 0.0
    # P = len(board_stack) # Number of Trotter slices
    #
    # for board in board_stack:
    #     energy += ClassicalEnergy(N, board, A, B, C)
    #
    # # Loop over Trotter slices
    # # for i in range(P):
    # #     for j in range(N):
    # #         for k in range(N - j):
    # #             energy += J_perp * # something

    # P = board_stack.shape[0]
    # E_classical = 0.0
    # # Sum classical energies for each slice.
    # for m in range(P):
    #     E_classical += ClassicalEnergy(N, board_stack[m], A, B, C)
    #
    # E_coupling = 0.0
    # # Sum coupling energies between adjacent slices (with periodic boundaries).
    # for m in range(P):
    #     next_m = (m + 1) % P
    #     E_coupling += np.sum(board_stack[m] * board_stack[next_m])
    #
    # total_energy = E_classical + J_perp * E_coupling

    classical_energy = 0.0
    coupling_energy = 0.0
    P = len(board_stack)  # Number of Trotter slices

    for board in board_stack:
        classical_energy += ClassicalEnergy(N, board, A, B, C)

    # Loop over Trotter slices, calculating effective energy
    for i in range(P):
        for j in range(N):
            for k in range(N - j):
                coupling_energy += J_perp * (2 * board_stack[i][j][k + j] - 1) * (2 * board_stack[i - 1][j][k + j] - 1)

    total_energy = classical_energy + coupling_energy

    return total_energy

def RunSimulatedAnnealing(N, monte_carlo_steps, annealing_steps, initial_T, final_T, A, B, C):

    board = GenerateBoard(N)

    temperature_range = np.linspace(initial_T, final_T, annealing_steps)

    energies = []

    CurrentEnergy = ClassicalEnergy(N, board, A, B, C)

    for T in temperature_range:
        for _ in range(monte_carlo_steps):

            NewBoard = SwapMove(board)[0]
            NewEnergy = ClassicalEnergy(N, NewBoard, A, B, C)

            # Apply Metroplis
            random_number = np.random.uniform()
            delta_E = NewEnergy - CurrentEnergy
            if delta_E < 0 or random_number < np.exp(-delta_E / T):
                board = NewBoard
                CurrentEnergy = NewEnergy

            energies.append(CurrentEnergy)

    return board, energies, CurrentEnergy

def RunPIQA(N, P, T, monte_carlo_steps, annealing_steps, initial_Gamma, final_Gamma, A, B, C):

    energies = []
    CurrentEnergy = np.inf

    # Copy the same board across all Trotter slices (as done by Martonak et. al.)
    board = GenerateBoard(N)
    board_stack = np.tile(board, (P, 1, 1))

    Gamma_schedule = np.linspace(initial_Gamma, final_Gamma, annealing_steps)

    classical_energies = [] # need these for plotting

    for step in range(annealing_steps):
        Gamma = Gamma_schedule[step]
        J_perpendicular = (-P*T / 2) * np.log(np.tanh(Gamma / (P*T)))

        # Perform several MC moves at this Gamma
        for _ in range(monte_carlo_steps):
            for slice in range(P):

                New_board_stack = QuantumSwapMove(board_stack, slice)
                NewEnergy = QuantumEnergy(N, New_board_stack, A, B, C, J_perpendicular)

                delta_E = NewEnergy - CurrentEnergy
                if delta_E < 0 or np.random.uniform() < np.exp(-delta_E / T):
                    board_stack = New_board_stack
                    CurrentEnergy = NewEnergy
                energies.append(CurrentEnergy)

                best_slice_energy = min(
                    ClassicalEnergy(N, board_stack[m], A, B, C) for m in range(P)
                )
                classical_energies.append(best_slice_energy)

    # Select the best slice from board_stack.
    FinalEnergy = ClassicalEnergy(N, board_stack[0], A, B, C)
    FinalBoard = board_stack[0]

    for i in range(1, P):
        energy_i = ClassicalEnergy(N, board_stack[i], A, B, C)
        if energy_i < FinalEnergy:
            FinalEnergy = energy_i
            FinalBoard = board_stack[i]

    return classical_energies, FinalEnergy, FinalBoard